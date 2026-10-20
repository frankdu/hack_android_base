# In-app model downloads

## Google AI Edge Gallery source review

Inspected Gallery revision `8f5f2a0bdb7c81693384dbdecbabec9b1e8d7f4e` on October 19, 2026:

- [DownloadRepository.kt](https://github.com/google-ai-edge/gallery/blob/8f5f2a0bdb7c81693384dbdecbabec9b1e8d7f4e/Android/src/app/src/main/java/com/google/ai/edge/gallery/data/DownloadRepository.kt): background work, per-model job IDs and progress observation.
- [DownloadWorker.kt](https://github.com/google-ai-edge/gallery/blob/8f5f2a0bdb7c81693384dbdecbabec9b1e8d7f4e/Android/src/app/src/main/java/com/google/ai/edge/gallery/worker/DownloadWorker.kt): foreground notifications, partial files, HTTP range requests, identity encoding and progress reports.
- [LiteRtLmModelDownloader.kt](https://github.com/google-ai-edge/gallery/blob/8f5f2a0bdb7c81693384dbdecbabec9b1e8d7f4e/Android/src/app/src/main/java/com/google/ai/edge/gallery/data/LiteRtLmModelDownloader.kt): file-backed model readiness and interrupted-download discovery.
- [HuggingFaceApiClient.kt](https://github.com/google-ai-edge/gallery/blob/8f5f2a0bdb7c81693384dbdecbabec9b1e8d7f4e/Android/src/app/src/main/java/com/google/ai/edge/gallery/huggingface/HuggingFaceApiClient.kt): host accessibility and file metadata.
- [1_0_20 model catalog](https://github.com/google-ai-edge/gallery/blob/8f5f2a0bdb7c81693384dbdecbabec9b1e8d7f4e/model_allowlists/1_0_20.json): the standard `gemma-4-E2B-it.litertlm` and `gemma-4-E4B-it.litertlm` files, with CPU/GPU support.

Our implementation uses the same official model repositories/files and a download-first flow. It is independently implemented; no Gallery source was copied. Gallery uses WorkManager and a custom HTTP worker. This small prototype delegates transfer lifecycle, network retries and notifications to Android's [DownloadManager](https://developer.android.com/reference/android/app/DownloadManager), avoiding an app-owned long-running service. File verification and registration remain on the app's serialized model worker.

## Pinned artifacts

| Slot | Repository | Revision | Bytes | SHA-256 |
|---|---|---|---:|---|
| E2B | `litert-community/gemma-4-E2B-it-litert-lm` | `41c5569623a7c189d6eec3e5225260c58cedbce7` | 2,588,147,712 | `181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c` |
| E4B | `litert-community/gemma-4-E4B-it-litert-lm` | `e6a3892a8c22cf87ff99583996149cf3cb32452b` | 3,659,530,240 | `0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0` |

File names match Gallery. Revisions are the official repositories' current commits when this change was made, rather than Gallery's older commits. Size and SHA-256 came from Hugging Face's public LFS metadata and were independently confirmed against the resolved download response headers. Both URLs redirected to HTTP 200 without authentication. The full multi-GB files were not downloaded on the development host.

`ModelCatalog.kt` is the single source of download URL/revision/size/hash data. No remote configuration, tokens, model hub login or third-party inference service is used.

## Lifecycle

1. Enqueue the chosen public URL with `INTERNET` permission and a destination inside `getExternalFilesDir("downloaded-models")`. No broad storage permission is needed.
2. Persist the Android download ID and immutable artifact spec in private preferences. Wi-Fi-only is enabled by default; roaming is disabled. The platform handles notifications and retry scheduling.
3. Observe Android transfer status and byte counts every second while this app process is alive. Transfers continue without that polling process. Re-query persisted IDs after restart.
4. On success, hash the file and compare exact size/SHA-256. Read native LiteRT-LM metadata. Reject truncated/corrupted/non-generative files.
5. Rename the verified file in the same directory and atomically update the model manifest. No import copy is made. Downloaded artifacts have `downloaded=true`; older imported models continue to resolve from their original private directory.
6. Remove the completed Android job after installation. If the process exits after rename but before manifest commit, the next verification can recover the installed candidate by its hash.

Inference and manual import are disabled while any download is active or verifying. This avoids simultaneous provisioning work affecting benchmark measurements and keeps native metadata inspection serialized with engine operations.

### Interruption and failure

Android can pause for network availability, Wi-Fi or retry backoff; these are waiting states rather than failed models. Automatic network/process interruption recovery belongs to the system transfer service. A permanently failed job exposes a Retry action. User cancellation removes the partial file; manual Retry starts fresh. We do not claim a user-controlled pause/resume feature.

Verification happens in the app process. Leaving the app does not cancel it, but Android can reclaim the process; unfinished verification is resumed as a fresh integrity check on next launch. A model is not ready until verification and manifest registration succeed. Failed/canceled downloads do not replace an installed model.

Downloaded models remain app-owned and are removed on uninstall. Clearing app data removes the registry and downloaded files. The app only contacts the model host/CDN for provisioning; prompts, outputs and benchmark results stay local unless explicitly exported.

## Validation boundaries

Unit tests cover pinned catalog values, long byte counters/progress bounds, correct SHA verification, truncated/missing file rejection, and corruption with an unchanged file length. Both full downloads, SHA-256/native metadata verification, model registration, and interruption/restart recovery were exercised successfully on a Pixel 10 Pro Fold running Android 17. Build/lint and host endpoint/range checks also passed. Notification interaction and inference performance are separate checks; see VALIDATION.md.
