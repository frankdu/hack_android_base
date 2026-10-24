# Public LiteRT model publication

All five pipelines were published as **public, ungated** Hugging Face models under
`frankdu` on 2026-10-24. All 274 uploaded files passed remote size and integrity
checks (LFS SHA-256 or Git blob hash), and anonymous graph downloads were verified
for every repository. See [machine-readable results](IMAGE_MODEL_PUBLICATION.json).

| Repository | Size | Verified files | Published revision |
|---|---|---|---|
| [frankdu/Bonsai-Image-ternary-4B-LiteRT](https://huggingface.co/frankdu/Bonsai-Image-ternary-4B-LiteRT) | 6.55 GB | 49 | `165657b39eaeeb4e1326b98bbfa19a49a895f5e0` |
| [frankdu/FLUX.2-klein-4B-LiteRT](https://huggingface.co/frankdu/FLUX.2-klein-4B-LiteRT) | 7.46 GB | 38 | `ce16220aacca7693f3b186466e18532d5ad646bb` |
| [frankdu/SD-Turbo-LiteRT](https://huggingface.co/frankdu/SD-Turbo-LiteRT) | 5.03 GB | 48 | `d4cc937ad9e51c8a78eb83dd9b99b7dbc1168dad` |
| [frankdu/SDXL-Turbo-LiteRT](https://huggingface.co/frankdu/SDXL-Turbo-LiteRT) | 13.76 GB | 96 | `7496c07ff0b70bc18441100b0b8b86f763cf8e8b` |
| [frankdu/iris-3b-Turbo-LiteRT](https://huggingface.co/frankdu/iris-3b-Turbo-LiteRT) | 13.71 GB | 43 | `ac493279ffff17ee9bcd12f79cda62864d3eff2c` |

SD-Turbo and SDXL-Turbo are new float32 LiteRT conversions, including model-specific
sampler graphs. FLUX, Bonsai and Iris mirror pinned upstream LiteRT conversions
with attribution. FLUX additionally includes the reconstructed CPU host loop and
fixed timestep/VAE constants. Model cards explain each package's provenance,
usage, licenses and validation limits.

Every repository contains runnable CPU examples, sample outputs, source
licenses/notices, validation records, an artifact manifest and SHA-256 checksums.
No credentials, caches, original Diffusers checkpoints or unrelated workstation
files were uploaded. SD/SDXL retain the Stability AI Community License, required
NOTICE and Powered by Stability AI attribution; public distribution does not
relicense their weights as Apache. Bonsai model and Android code licenses are
included separately.

## Download and use

Download a published repository with the Hugging Face CLI, for example:

```sh
hf download frankdu/SD-Turbo-LiteRT --local-dir SD-Turbo-LiteRT
cd SD-Turbo-LiteRT
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run_portable.py --model-dir . --prompt "a blue ceramic teapot on a wooden table" --seed 7 --out output.png
```

Use each repository's README for its own runner and model-directory layout. The
Android app's existing community entries still use their pinned upstream sources;
its SD/SDXL LiteRT entries still use the staged portable-bundle installation path.
Publication alone does not change the app's download catalog.

## Validation

All five pipelines completed desktop LiteRT CPU generation checks. The packaged
SD/SDXL scripts additionally passed a new teapot prompt with seed 7, the packaged
FLUX runner completed its apple generation, and the Java noise helper matched the
JVM stream bit for bit. These are desktop tests; Android CPU/GPU, phone memory
requirements and Tensor TPU execution remain unvalidated.

## Workstation and repeatable publication

The source packages remain under
`192.168.4.103:/mnt/sda1/frankdu/litert-image-support/publications/`. Preparation
uses hardlinks on the secondary disk and preserves original verified bundles.
The initial attempt was rejected because both logins were read-only; publication
succeeded after the workstation login was updated with write access.

Preparation, final verification and upload helpers are in
`scripts/image_litert/{prepare_publication,finalize_publication,upload_publication}.py`.
The uploader requires explicit `--upload`, permits only the prepared file
whitelist, preserves existing repository visibility and verifies the results.
To repeat an unchanged publication from the workstation:

```sh
cd /mnt/sda1/frankdu/litert-image-support
. venv/bin/activate
export HF_HOME=/mnt/sda1/frankdu/litert-image-support/hf
export HF_TOKEN_PATH=/home/frankdu/.cache/huggingface/token
python upload_publication.py --root "$PWD" --namespace frankdu --upload
```

Token contents are never printed or supplied as command-line arguments. The
uploader writes exact revisions and verification results to
`publication-results.json`. `run_portable.py` uses LiteRT Interpreter CPU
signatures for int64 I/O; Android uses CompiledModel and TensorBuffer.writeLong.
