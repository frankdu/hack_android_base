"""Run with: python -m unittest discover -s scripts -p 'test_generate_word_images.py'.

Uses real Pillow encoding and a fake model; no torch installation/model downloads.
"""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import generate_word_images as generation


class Generator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class Flux2KleinPipeline:
    vae_scale_factor = 8

    def __init__(self):
        self.calls = []
        self.fail_on = None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_on:
            raise RuntimeError("simulated generation failure")
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), "green")])


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dictionary = self.root / "dictionary.txt"
        self.dictionary.write_text("Category: animal\ndog\ncat\ncow\n", encoding="utf-8")
        self.output = self.root / "images"
        self.pipeline = Flux2KleinPipeline()
        self.torch = SimpleNamespace(Generator=Generator, inference_mode=nullcontext)
        self.loader = self.enterContext(patch.object(generation, "load_pipeline", return_value=(self.pipeline, self.torch)))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))

    def args(self, *extra):
        return generation.parse_args(["--dictionary", str(self.dictionary), "--output-dir", str(self.output), *extra])

    def test_project_dictionary_and_unique_filenames(self):
        entries = generation.read_dictionary(generation.ROOT / "seed/kids_english_dictionary.txt")
        self.assertEqual(len(entries), 10000)
        self.assertEqual(len({category for _, category in entries}), 100)
        self.assertEqual(len({generation.output_name(*entry, "png").casefold() for entry in entries}), 10000)
        self.assertEqual(generation.output_name("a/b: c?", "sea creature", "jpg"), "a-b-c_sea-creature.jpg")

    def test_limit_resume_and_skip_without_loading_model(self):
        self.assertEqual(generation.run(self.args("--limit", "1")), 0)
        first_bytes = (self.output / "dog_animal.png").read_bytes()
        self.assertEqual(len(self.pipeline.calls), 1)
        self.assertEqual(self.pipeline.calls[0]["width"], 528)
        with Image.open(self.output / "dog_animal.png") as image:
            self.assertEqual(image.size, (520, 520))
        self.assertEqual(generation.run(self.args()), 0)
        self.assertEqual(len(self.pipeline.calls), 3)
        self.assertEqual((self.output / "dog_animal.png").read_bytes(), first_bytes)
        self.loader.reset_mock()
        self.assertEqual(generation.run(self.args()), 0)
        self.loader.assert_not_called()

    def test_failure_then_resume_retains_seeds(self):
        self.pipeline.fail_on = 2
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            generation.run(self.args())
        self.assertTrue((self.output / "dog_animal.png").is_file())
        self.assertFalse((self.output / "cat_animal.png").exists())
        failed_seed = self.pipeline.calls[1]["generator"].seed
        self.pipeline.fail_on = None
        self.assertEqual(generation.run(self.args()), 0)
        self.assertEqual(self.pipeline.calls[2]["generator"].seed, failed_seed)
        self.assertEqual(len(list(self.output.glob("*.png"))), 3)

    def test_progress_total_excludes_cached_images_and_respects_limit(self):
        from tqdm import tqdm

        bars = []

        def make_bar(**kwargs):
            bar = tqdm(**kwargs, file=io.StringIO(), mininterval=0)
            bars.append(bar)
            return bar

        with patch("tqdm.tqdm", side_effect=make_bar) as factory:
            generation.run(self.args("--limit", "1"))
            self.assertEqual((bars[-1].n, bars[-1].total), (1, 1))
            generation.run(self.args())
            self.assertEqual((bars[-1].n, bars[-1].total), (2, 2))
            factory.reset_mock()
            generation.run(self.args())
            factory.assert_not_called()

    def test_progress_counts_failed_attempts_when_continuing(self):
        from tqdm import tqdm

        bar = tqdm(total=3, file=io.StringIO(), mininterval=0)
        self.pipeline.fail_on = 2
        with patch("tqdm.tqdm", return_value=bar):
            self.assertEqual(generation.run(self.args("--continue-on-error")), 1)
        self.assertEqual((bar.n, bar.total), (3, 3))
        self.assertIn("failed=1", bar.postfix)
        self.assertEqual(len(list(self.output.glob("*.png"))), 2)

    def test_corrupt_and_wrong_size_images_regenerated(self):
        self.output.mkdir()
        (self.output / "dog_animal.png").write_bytes(b"broken")
        Image.new("RGB", (8, 8)).save(self.output / "cat_animal.png")
        Image.new("RGB", (520, 520)).save(self.output / "cow_animal.png")
        generation.run(self.args())
        self.assertEqual(len(self.pipeline.calls), 2)
        for path in self.output.glob("*.png"):
            self.assertTrue(generation.existing_image(path, (520, 520), "png"))

    def test_jpeg_and_continue_after_failure(self):
        self.pipeline.fail_on = 1
        self.assertEqual(generation.run(self.args("--format", "jpg", "--width", "523", "--height", "517", "--continue-on-error")), 1)
        self.assertFalse((self.output / "dog_animal.jpg").exists())
        for path in self.output.glob("*.jpg"):
            with Image.open(path) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (523, 517))
        self.assertEqual(len(list(self.output.glob("*.jpg"))), 2)

    def test_interrupted_save_keeps_previous_file_and_removes_temporary(self):
        self.output.mkdir()
        path = self.output / "dog_animal.png"
        generation.save_image(Image.new("RGB", (520, 520)), path, (520, 520), "png")
        before = path.read_bytes()
        with patch.object(Image.Image, "save", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                generation.save_image(Image.new("RGB", (520, 520)), path, (520, 520), "png")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.output.iterdir()), [path])

    def test_dry_run_has_no_writes_and_collision_fails_before_loading(self):
        self.assertEqual(generation.run(self.args("--dry-run")), 0)
        self.assertFalse(self.output.exists())
        self.loader.assert_not_called()
        self.dictionary.write_text("Category: animal\na/b\na:b\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Filename collision"):
            generation.run(self.args())
        self.loader.assert_not_called()

    def test_filters_and_malformed_dictionary(self):
        generation.run(self.args("--category", "ANIMAL", "--word", "cat"))
        self.assertEqual([path.name for path in self.output.iterdir()], ["cat_animal.png"])
        with self.assertRaisesRegex(ValueError, "Unknown category"):
            generation.run(self.args("--category", "typo"))
        for text in ("dog\n", "Category:\ndog\n", "Category: animal\n", ""):
            self.dictionary.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ValueError):
                generation.read_dictionary(self.dictionary)


class ModelLoadingTests(unittest.TestCase):
    def test_auto_moves_pipeline_to_available_gpu_before_cpu(self):
        from unittest.mock import MagicMock

        for cuda, mps, expected in ((True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")):
            with self.subTest(cuda=cuda, mps=mps):
                torch = SimpleNamespace(
                    cuda=SimpleNamespace(is_available=lambda: cuda, is_bf16_supported=lambda: True),
                    backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
                    bfloat16="bf16", float32="fp32",
                )
                pipeline_type = MagicMock()
                args = generation.parse_args([])
                with patch.dict("sys.modules", {"torch": torch, "diffusers": SimpleNamespace(DiffusionPipeline=pipeline_type)}), redirect_stdout(io.StringIO()):
                    pipeline, _ = generation.load_pipeline(args)
                pipeline.to.assert_called_once_with(expected)
                pipeline.enable_model_cpu_offload.assert_not_called()

    def test_explicit_unavailable_gpu_fails_before_model_download(self):
        from unittest.mock import MagicMock

        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        pipeline_type = MagicMock()
        for device in ("cuda", "mps"):
            with self.subTest(device=device):
                args = generation.parse_args(["--device", device])
                with patch.dict("sys.modules", {"torch": torch, "diffusers": SimpleNamespace(DiffusionPipeline=pipeline_type)}):
                    with self.assertRaisesRegex(ValueError, "unavailable"):
                        generation.load_pipeline(args)
        pipeline_type.from_pretrained.assert_not_called()

    def test_gpu_initialization_failure_does_not_silently_fall_back_to_cpu(self):
        from unittest.mock import MagicMock

        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True), bfloat16="bf16")
        pipeline_type = MagicMock()
        pipeline_type.from_pretrained.return_value.to.side_effect = RuntimeError("GPU out of memory")
        with patch.dict("sys.modules", {"torch": torch, "diffusers": SimpleNamespace(DiffusionPipeline=pipeline_type)}), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "GPU out of memory"):
                generation.load_pipeline(generation.parse_args([]))
        pipeline_type.from_pretrained.return_value.to.assert_called_once_with("cuda")

    def test_download_cache_offline_and_cpu_offload_configuration(self):
        from unittest.mock import MagicMock

        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True), bfloat16="bf16")
        pipeline_type = MagicMock()
        args = generation.parse_args(["--model", "example/model", "--cache-dir", "/tmp/model-cache", "--revision", "abc", "--offline", "--cpu-offload"])
        with patch.dict("sys.modules", {"torch": torch, "diffusers": SimpleNamespace(DiffusionPipeline=pipeline_type)}), redirect_stdout(io.StringIO()):
            pipeline, _ = generation.load_pipeline(args)
        pipeline_type.from_pretrained.assert_called_once_with(
            "example/model", torch_dtype="bf16", local_files_only=True,
            cache_dir="/tmp/model-cache", revision="abc",
        )
        pipeline.enable_model_cpu_offload.assert_called_once_with()
        pipeline.to.assert_not_called()


if __name__ == "__main__":
    unittest.main()
