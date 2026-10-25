"""Experimental 512px exports of the three exact Diffusers checkpoints.

Each neural component is exported once and split into weight-bounded graphs.
Scheduler/latent transforms are exported too, so Android never reimplements a
model-specific sampler. Fixed resolution and default steps are intentional.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from graph_export import ProgramBuilder
from workflow import digest, write_json


class Clip(torch.nn.Module):
    def __init__(self, encoder, xl=False):
        super().__init__(); self.encoder, self.xl = encoder, xl
    def forward(self, tokens):
        out = self.encoder(tokens, output_hidden_states=self.xl, return_dict=True)
        if self.xl:
            return out.hidden_states[-2], out[0]
        return out.last_hidden_state


class Qwen(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.encoder = encoder.model  # no unused vocabulary projection
    def forward(self, tokens, mask):
        out = self.encoder(input_ids=tokens, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True)
        return torch.stack([out.hidden_states[k] for k in (9, 18, 27)], dim=1).permute(0, 2, 1, 3).reshape(1, 512, 7680)


class Denoiser(torch.nn.Module):
    def __init__(self, unet, xl=False):
        super().__init__(); self.unet, self.xl = unet, xl
    def forward(self, latent, timestep, context, pooled=None, time_ids=None):
        extra = dict(text_embeds=pooled, time_ids=time_ids) if self.xl else None
        return self.unet(latent, timestep, encoder_hidden_states=context, added_cond_kwargs=extra, return_dict=False)[0]


class FluxDenoiser(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__(); self.transformer = transformer
    def forward(self, latent, timestep, context, text_ids, image_ids):
        return self.transformer(hidden_states=latent, timestep=timestep / 1000,
            encoder_hidden_states=context, txt_ids=text_ids, img_ids=image_ids,
            guidance=None, return_dict=False)[0]


class Cat(torch.nn.Module):
    def forward(self, x, y):
        return torch.cat([x, y], dim=-1)


class Scale(torch.nn.Module):
    def __init__(self, scale):
        super().__init__(); self.register_buffer("scale", torch.tensor(float(scale)))
    def forward(self, x):
        return x * self.scale


class AncestralStep(torch.nn.Module):
    def __init__(self, sigma, next_sigma, prediction):
        super().__init__()
        self.sigma, self.prediction = float(sigma), prediction
        up = torch.sqrt(next_sigma**2 * (sigma**2 - next_sigma**2) / sigma**2)
        self.up, self.down = float(up), float(torch.sqrt(next_sigma**2 - up**2))
    def forward(self, latent, prediction, noise):
        if self.prediction == "v_prediction":
            original = prediction * (-self.sigma / (self.sigma**2 + 1)**0.5) + latent / (self.sigma**2 + 1)
        elif self.prediction == "epsilon":
            original = latent - self.sigma * prediction
        else:
            raise ValueError(f"Unsupported scheduler prediction type: {self.prediction}")
        derivative = (latent - original) / self.sigma
        return latent + derivative * (self.down - self.sigma) + noise * self.up


class FlowStep(torch.nn.Module):
    def __init__(self, dsigma):
        super().__init__(); self.dsigma = float(dsigma)
    def forward(self, latent, prediction):
        return latent + self.dsigma * prediction


class Decode(torch.nn.Module):
    def __init__(self, vae, flux=False):
        super().__init__(); self.vae, self.flux = vae, flux
    def forward(self, latent):
        if self.flux:
            # Packed [1,1024,128] -> normalized 128-channel 32x32 -> unpatchified 32x64x64.
            latent = latent.permute(0, 2, 1).reshape(1, 128, 32, 32)
            latent = latent * torch.sqrt(self.vae.bn.running_var.reshape(1, -1, 1, 1) + self.vae.config.batch_norm_eps) + self.vae.bn.running_mean.reshape(1, -1, 1, 1)
            latent = latent.reshape(1, 32, 2, 2, 32, 32).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, 64, 64)
        else:
            latent = latent / self.vae.config.scaling_factor
        return (self.vae.decode(latent, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)


def save_tokenizer(builder, tokenizer, index, prompt, flux=False):
    folder = builder.out / f"tokenizer{index}"
    tokenizer.save_pretrained(folder)
    # Export a deliberately small, explicit byte-BPE ABI and fixtures. Android validates
    # these against the Python tokenizer before permitting arbitrary prompts.
    if flux:
        before = tokenizer.apply_chat_template([dict(role="user", content="WORD_IMAGE_SENTINEL")],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix, suffix = before.split("WORD_IMAGE_SENTINEL")
    else:
        prefix = suffix = ""
    vocab = tokenizer.get_vocab()
    # tokenizers JSON includes added-token IDs and merge ordering without requiring JNI.
    if (folder / "tokenizer.json").is_file():
        fast = json.loads((folder / "tokenizer.json").read_text())
        merges = fast["model"]["merges"]
        merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in merges]
    else:
        merges = [line for line in (folder / "merges.txt").read_text().splitlines() if line and not line.startswith("#")]
    spec = dict(kind="qwen" if flux else "clip", vocab=vocab, merges=merges,
        length=512 if flux else 77, pad=tokenizer.pad_token_id,
        bos=None if flux else tokenizer.bos_token_id, eos=None if flux else tokenizer.eos_token_id,
        prefix=prefix, suffix=suffix, paddingSide=tokenizer.padding_side,
        specialTokens=[dict(text=t, id=i) for t, i in tokenizer.get_added_vocab().items()])
    if flux:
        pre = fast["pre_tokenizer"]
        splits = pre.get("pretokenizers", [pre])
        spec["pattern"] = next(s["pattern"]["Regex"] for s in splits if s["type"] == "Split")
    fixtures = []
    for text in [prompt, "a red apple", "happy child's smile!", "café & blue circles", "one bicycle, two wheels", "surprised 😊", "a" * 800]:
        result = tokenizer(prefix + text + suffix, padding="max_length", max_length=spec["length"], truncation=True, return_tensors="pt")
        fixtures.append(dict(prompt=text, ids=result.input_ids[0].tolist(), mask=result.attention_mask[0].tolist()))
    spec["fixtures"] = fixtures
    relative = f"tokenizer{index}/bpe.json"
    write_json(builder.out / relative, spec)
    builder.files.add(relative)
    result = tokenizer(prefix + prompt + suffix, padding="max_length", max_length=spec["length"], truncation=True, return_tensors="pt")
    tokens = builder.external("tokens", result.input_ids.to(torch.int64), tokenizer=relative)
    mask = builder.external("mask", result.attention_mask.to(torch.int64), tokenizer=relative) if flux else None
    return tokens, mask


def export_pipeline(args):
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
    source = Path(args.source)
    meta = json.loads((source / "source.json").read_text())
    if meta["format"] != "diffusers":
        raise ValueError("Use an original Diffusers checkpoint for an app-compatible 512px export. The community FLUX 256px graphs can be AOT-tested with compile directly.")
    model, flux = meta["model"], meta["model"] == "FLUX_KLEIN"
    steps = 1 if model == "SD_TURBO" else 4
    pipe = DiffusionPipeline.from_pretrained(str(source), torch_dtype=torch.float32, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    # Eager attention avoids a fused attention operator that may be unsupported by export.
    for name in ("text_encoder", "text_encoder_2"):
        encoder = getattr(pipe, name, None)
        if encoder is not None:
            encoder.eval()
            if hasattr(encoder, "set_attn_implementation"):
                encoder.set_attn_implementation("eager")
    builder = ProgramBuilder(args.output, args.shard_mib)
    prompt = "One ripe red apple with a green leaf, clear educational illustration, plain cream background."
    tokens, mask = save_tokenizer(builder, pipe.tokenizer, 0, prompt, flux)
    generator = torch.Generator().manual_seed(42)
    if flux:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
        context = builder.call("qwen", Qwen(pipe.text_encoder), [tokens, mask], "encode")[0]
        latent_value, image_ids = pipe.prepare_latents(1, pipe.transformer.config.in_channels // 4,
            512, 512, torch.float32, torch.device("cpu"), generator)
        latent = builder.external("normal", latent_value)  # initial distribution is N(0,1)
        text_ids = builder.constant(pipe._prepare_text_ids(builder.values[context]))
        image_ids = builder.constant(image_ids)
        sigmas = np.linspace(1.0, 1 / steps, steps)
        if getattr(pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        timesteps, _ = retrieve_timesteps(pipe.scheduler, steps, torch.device("cpu"), sigmas=sigmas,
            mu=compute_empirical_mu(image_seq_len=latent_value.shape[1], num_steps=steps))
        for i, time in enumerate(timesteps):
            t = builder.constant(time.reshape(1).float())
            prediction = builder.call("dit", FluxDenoiser(pipe.transformer), [latent, t, context, text_ids, image_ids], "denoise", i + 1)[0]
            latent = builder.call(f"flow{i}", FlowStep(pipe.scheduler.sigmas[i + 1] - pipe.scheduler.sigmas[i]), [latent, prediction], "denoise", i + 1)[0]
    else:
        xl = model == "SDXL_TURBO"
        context_outputs = builder.call("clip0", Clip(pipe.text_encoder, xl), [tokens], "encode")
        context, pooled = context_outputs[0], None
        if xl:
            tokens2, _ = save_tokenizer(builder, pipe.tokenizer_2, 1, prompt)
            context2, pooled = builder.call("clip1", Clip(pipe.text_encoder_2, True), [tokens2], "encode")
            context = builder.call("context", Cat(), [context, context2], "encode")[0]
        # Same scheduler configuration as the source Turbo pipeline, including timestep spacing.
        scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        scheduler.set_timesteps(steps)
        noise_shape = (1, pipe.unet.config.in_channels, 64, 64)
        noise = builder.external("normal", torch.randn(noise_shape, generator=generator))
        latent = builder.call("initial", Scale(scheduler.init_noise_sigma), [noise], "encode")[0]
        time_ids = builder.constant(torch.tensor([[512, 512, 0, 0, 512, 512]], dtype=torch.float32)) if xl else None
        for i, time in enumerate(scheduler.timesteps):
            scaled = builder.call(f"scale{i}", Scale(1 / torch.sqrt(scheduler.sigmas[i]**2 + 1)), [latent], "denoise", i + 1)[0]
            t = builder.constant(time.reshape(1).float())
            inputs = [scaled, t, context] + ([pooled, time_ids] if xl else [])
            prediction = builder.call("unet", Denoiser(pipe.unet, xl), inputs, "denoise", i + 1)[0]
            noise = builder.external("normal", torch.randn(noise_shape, generator=generator))
            latent = builder.call(f"step{i}", AncestralStep(scheduler.sigmas[i], scheduler.sigmas[i + 1], scheduler.config.prediction_type), [latent, prediction, noise], "denoise", i + 1)[0]
    pixels = builder.call("vae", Decode(pipe.vae, flux), [latent], "decode")[0]
    from PIL import Image
    image = builder.values[pixels][0].permute(1, 2, 0).numpy()
    Image.fromarray((image * 255).round().astype(np.uint8)).save(builder.out / "reference.png")
    builder.files.add("reference.png")
    # Copy the original notices, not a newly invented permissive license.
    for path in source.iterdir():
        if path.is_file() and any(term in path.name.lower() for term in ("license", "notice", "policy")):
            shutil.copyfile(path, builder.out / path.name)
            builder.files.add(path.name)
    if (source / "README.md").is_file():
        shutil.copyfile(source / "README.md", builder.out / "SOURCE_MODEL_CARD.md")
        builder.files.add("SOURCE_MODEL_CARD.md")
    manifest = dict(format="word-image-litert-v1", modelId=model, source=meta, width=512, height=512,
        steps=steps, compiled=False, targetSoc=None, validation="cpu-shard-parity-only",
        externals=builder.externals, actions=builder.actions, output=pixels,
        files={name: dict(bytes=(builder.out / name).stat().st_size, sha256=digest(builder.out / name)) for name in sorted(builder.files)})
    from tensor_aliases import annotate_manifest
    annotate_manifest(builder.out, manifest)
    write_json(builder.out / "manifest.json", manifest)
    print(f"Exported {len(builder.files)} files. Next: compile separately for Tensor_G5 / Tensor_G6, then validate on-device.")
