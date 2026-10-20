# Validation status

Initial implementation: October 19, 2026.

## Completed checks

`./gradlew --no-daemon :app:assembleDebug :app:testDebugUnitTest :app:lintDebug` completed successfully on the final implementation.

- **Debug APK built:** `app/build/outputs/apk/debug/app-debug.apk`.
- **13 unit tests passed**, no failures/errors/skips. Coverage includes warm-up exclusion, stable batch IDs, partial results on failure, stop behavior, unavailable/non-finite metrics, parameter validation, CSV quoting/formula escaping, the pinned download catalog, long byte counters, valid file verification, truncation/missing file rejection and same-size corruption.
- **Android lint passed with no errors.** Remaining notices concern newer dependency versions, intentional ARM64-only packaging, and a KTX style suggestion. Pinned versions were retained after successful build verification.
- **APK manifest checked with `aapt`:** application `dev.gemmabench`, min SDK 31, target/compile SDK 37, launcher `MainActivity`, ARM64 native libraries, and `INTERNET` permission for model provisioning. Inference remains local.
- **Shell syntax checked:** `sh -n scripts/push-model.sh`.
- LiteRT-LM API usage was checked against the actual pinned 0.18.0 Maven AAR, including `benchmark()`, `BenchmarkInfo`, `ModelInfo`, `ConversationConfig` and streaming conversation APIs.

## Pixel download validation

A **Pixel 10 Pro Fold, Tensor G5, Android 17** became available during the download feature implementation. The updated debug APK was installed successfully.

- Both real model transfers were started using the app's controller and Android DownloadManager with Wi-Fi-only enabled.
- The app process was intentionally stopped during transfer. After restart, the persisted Android job records were unchanged and downloading continued. The recovery smoke instrumentation test passed.
- **The full physical-device download/install instrumentation test passed:** E2B (2,588,147,712 bytes) and E4B (3,659,530,240 bytes) completed, matched their pinned SHA-256 checksums, passed native LiteRT-LM metadata inspection and were registered as ready models.
- The device UI was captured and visually inspected, including a ready E2B card and E4B download progress/cancel control. The final app retains both downloaded models across restart.
- The public pinned URLs were also checked from the host: redirects ended at HTTP 200 with matching linked size/hash metadata; bounded 1,024-byte range requests succeeded for both. No full host-side model copies were made.

Instrumentation tests are opt-in because they can provision about 6.25 GB. Build/install `app-debug-androidTest.apk`, then run:

```sh
adb shell am instrument -w \
  -e class dev.gemmabench.ModelDownloadInstrumentedTest \
  -e downloadModels true \
  dev.gemmabench.test/androidx.test.runner.AndroidJUnitRunner
```

Add `-e downloadSmokeOnly true` for a short enqueue/recovery/received-byte check. Existing tracked or correctly installed catalog models are reused.

## Benchmark/device checks still required

No model-generation benchmark numbers have been measured by this project yet. Online reference numbers are in RESEARCH_0.md and are not seeded into app results. GPU/TPU inference, generation quality, benchmark cancellation and peak memory behavior still need their own device runs.

On a physical Pixel:

1. Open Models and download E2B/E4B (already completed on the connected test Pixel). Confirm ready state and detected metadata under Advanced.
2. Run CPU 256/128 with one warm-up and three measured runs; export JSON.
3. Repeat with GPU-compatible artifacts and identical workload, allowing the phone to cool.
4. Run the same short prompt twice on one model to compare engine loading against reuse. Check response quality and exact native token counts.
5. Rotate during a run, stop a prompt, background the app during a benchmark, and confirm completed results persist across restart.
6. If provisioning TPU libraries, use matching G5/G6 artifacts and firmware and repeat independently. Report unsupported initialization rather than substituting CPU/GPU results.
