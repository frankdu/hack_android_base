# LiteRT image pipeline sources

- `app/src/main/java/dev/gemmabench/imagegen/iris/` derives from the Android reference in
  [litert-community/iris-3b-Turbo-LiteRT](https://huggingface.co/litert-community/iris-3b-Turbo-LiteRT),
  revision `743fd20acefd7f9ddcf3585f88bcda3257f7213a`. See `IRIS_LICENSE` and `IRIS_NOTICE`.
  Changes: package relocation, configurable tokenizer added-token filename, and direct
  embedding-file access instead of symlinks (Android external storage cannot create symlinks).
  The app adapter forces the selected CPU/GPU backend, FP32 GPU precision and sequential
  residency, checks finite outputs, and records progress and timings.
- `app/src/main/java/dev/gemmabench/imagegen/bonsai/BonsaiMath.kt` and `QwenTokenizer.kt`
  derive from [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B),
  revision `317d577f5f962eb2d5a3980bca2ce047a8338212`. See `BONSAI_LICENSE` and `BONSAI_NOTICE.md`.
  Changes: package relocation and Gson instead of Android JSONObject for JVM testing.
- The compressed tokenizer test data comes from these repositories and the published
  [FLUX LiteRT tokenizer](https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT),
  revision `223f4c989d8e5275fa8e695ddebea8d7ac7bada6`. Tokenizer tables derive from
  Qwen (Alibaba Cloud, Apache 2.0). They contain no model weights.
- `app/src/main/assets/image-litert/flux_host.json` contains fixed timestep embeddings
  and VAE normalization statistics derived from Black Forest Labs' Apache-2.0
  [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), revision
  `e7b7dc27f91deacad38e78976d1f2b499d76a294`. Reproduce it with
  `scripts/image_litert/prepare_flux_host.py`. The app's separate FLUX host implementation
  follows the published graph ABI and upstream Diffusers mathematics.

The Android APK also carries these license/notice files in `assets/image-litert/licenses/`.
Model graph weights are downloaded separately, and keep their source licenses.
