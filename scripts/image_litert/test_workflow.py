import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from graph_export import ProgramBuilder, natural_names
from export_pipeline import AncestralStep, FlowStep
from workflow import compile_bundle, digest, parser, publish


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)
        self.b = torch.nn.Linear(4, 4)
    def forward(self, x):
        y = self.a(x)
        return self.b(y) + x, y


class WorkflowTest(unittest.TestCase):
    def test_shards_bind_weights_reuse_files_and_preserve_multiple_outputs(self):
        calls = []
        def converter(module, args, path, expected):
            # A real conversion needs the Linux toolchain; this test isolates FX planning.
            self.assertTrue(all(v.shape == (1, 4) for v in args))
            self.assertTrue(list(module.buffers()))  # weights embedded, not runtime arguments
            path.write_bytes(b"test-only")
            calls.append(path)
            return dict(signature="serving_default", inputNames=[f"args_{i}" for i in range(len(args))],
                        outputNames=[f"output_{i}" for i in range(len(expected))], maxAbsErrors=[0] * len(expected))
        with tempfile.TemporaryDirectory() as folder:
            b = ProgramBuilder(folder, shard_mib=0.0001, converter=converter)
            module = Toy().eval()
            x = b.external("normal", torch.ones(1, 4))
            outputs = b.call("toy", module, [x], "encode")
            for reference, want in zip(outputs, module(b.values[x])):
                torch.testing.assert_close(b.values[reference], want)
            count = len(calls)
            x2 = b.external("normal", torch.full((1, 4), 2.0))
            outputs = b.call("toy", module, [x2], "denoise", 1)
            self.assertEqual(len(calls), count)
            self.assertGreater(count, 1)
            for reference, want in zip(outputs, module(b.values[x2])):
                torch.testing.assert_close(b.values[reference], want)
            available = {e["tensor"] for e in b.externals}
            for action in b.actions:
                self.assertTrue(all(i["tensor"] in available for i in action["inputs"]))
                available.update(o["tensor"] for o in action["outputs"])

    def test_natural_tensor_order_and_unknown_abi(self):
        self.assertEqual(natural_names(["args_10", "args_2", "args_0"]), ["args_0", "args_2", "args_10"])
        with self.assertRaises(ValueError):
            natural_names(["changed_api"])

    def test_euler_ancestral_matches_source_scheduler(self):
        from diffusers import EulerAncestralDiscreteScheduler
        for prediction in ("epsilon", "v_prediction"):
            scheduler = EulerAncestralDiscreteScheduler(prediction_type=prediction, timestep_spacing="trailing")
            scheduler.set_timesteps(4)
            generator = torch.Generator().manual_seed(7)
            latent = torch.randn(1, 4, 4, 4)
            for i, t in enumerate(scheduler.timesteps):
                noise_generator = torch.Generator().manual_seed(7 + i)
                prediction_value = torch.full_like(latent, 0.2)
                noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(7 + i))
                scheduler.scale_model_input(latent, t)
                want = scheduler.step(prediction_value, t, latent, generator=noise_generator).prev_sample
                got = AncestralStep(scheduler.sigmas[i], scheduler.sigmas[i + 1], prediction)(latent, prediction_value, noise)
                torch.testing.assert_close(got, want)
                latent = want

    def test_flow_euler(self):
        torch.testing.assert_close(FlowStep(-0.25)(torch.ones(2), torch.ones(2) * 2), torch.full((2,), 0.5))

    def test_compile_rejects_unsupported_host_before_loading_sdk(self):
        args = parser().parse_args(["compile", "--source", "missing", "--output", "missing", "--soc", "Tensor_G5"])
        with patch("platform.system", return_value="Darwin"), self.assertRaises(SystemExit):
            compile_bundle(args)

    def test_publishing_is_dry_by_default_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph = root / "test.tflite"; graph.write_bytes(b"test")
            m = dict(compiled=True, modelId="FLUX_KLEIN", targetSoc="Tensor_G5", steps=4,
                     validation="not-device-validated", source=dict(revision="abc"),
                     files={"test.tflite": dict(sha256=digest(graph))})
            (root / "manifest.json").write_text(json.dumps(m))
            args = parser().parse_args(["publish", "--source", folder])
            publish(args)
            self.assertIn("not-device-validated", (root / "README.md").read_text())
            graph.write_bytes(b"changed")
            with self.assertRaises(SystemExit):
                publish(args)


if __name__ == "__main__":
    unittest.main()
