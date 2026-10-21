# People across photos

Open **Photos → Find people**, grant image access, and tap **Find people in next batch**. Each eligible still photo is processed locally with YuNet face detection and SFace embeddings. Gemma need not be installed. Detected screenshots and unsupported image formats use the same exclusions as object analysis. Videos and hidden-face body matching are outside this feature.

The People gallery shows unnamed person groups and distinct photo counts. Open a group to name it, merge it into another group, or move an incorrectly grouped face to an existing/new group. The yellow rectangle identifies the face being moved in a group photo. Names are user supplied. Small detected faces without a usable embedding appear under Unassigned faces and can be assigned manually.

Stop/Resume retains pending batch items. Start over resets pagination while preserving cached results, identities, names, and corrections. Failed photos retry when scanned again. Clear person data deletes face embeddings, group records, names, corrections, and face-analysis cache entries; it leaves source photos and object analyses intact. Existing object JSON exports do not include embeddings or person records.

## Models and matching

- Android runtime: `org.opencv:opencv:4.13.0`, OpenCV DNN CPU backend, arm64, independent of LiteRT-LM.
- Detector: `face_detection_yunet_2023mar.onnx`, 232,589 bytes, SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- Recognizer: `face_recognition_sface_2021dec.onnx`, 38,696,353 bytes, SHA-256 `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`.
- Source: [OpenCV Zoo revision 48b10fa505aeb51fbd127fc60a5c660040e823b7](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7). Model SHA-256 values also match the upstream Git LFS pointers. YuNet's directory declares MIT licensing; SFace's declares Apache 2.0. Both license texts ship alongside the models in `assets/people`.

The pipeline decodes the upright image with EXIF orientation applied, bounded to 2048 pixels on its longest side, one photo at a time. YuNet detects on a second image bounded to 640 pixels, with detection score 0.9 and NMS 0.3. Five landmarks map back to the larger source for SFace alignment. Faces smaller than 48 pixels on the decoded source's shorter face-box side remain unassigned. Embeddings are L2 normalized; cosine similarity must reach **0.39** and exceed the next person's score by **0.10** to link a face automatically.

Each person retains one fixed anchor face. New faces cannot gradually shift the anchor through a chain of increasingly dissimilar matches. Uncertain usable faces create separate unnamed groups; users can merge these. Existing groups are never automatically merged. Moving a face locks its assignment and records its rejected former group. Merging retains the selected target ID and name, and records source-ID aliases. Unchanged cached photos preserve face IDs; source edits trigger new observations and may require corrections again.

Room schema 2 adds person, face, per-photo status/cache, and correction tables through an explicit migration from schema 1. The existing object cache is preserved. Cache keys include source revision, preprocessing version, and model hashes. Embeddings also carry a model/preprocessing identity so incompatible vectors cannot be compared. The gallery and matching anchors are filtered against the current permission-scoped MediaStore inventory and source revisions, hiding revoked/deleted/edited photos. Reopening People refreshes access and restarts pagination.

All photos and vectors stay on the phone; no recognition network requests occur. Bundled model copies are verified before native loading and stored in the app's no-backup directory. The app already disables backup. Native calls finish before a requested stop takes effect; an unfinished photo is not cached.

## Calibration and measured limits

The bundled `assets/people/calibration.json` records a deterministic LFW-funneled experiment: 40 identities, eight photos each, seed 42, split into 20 calibration and 20 identity-disjoint held-out identities. The preparation script keeps the images under ignored `artifacts/`; they are not shipped. Largest-face selection uses LFW's centered-face convention.

On a Pixel 10 Pro Fold:

| Measurement | Result |
|---|---|
| Photos with a detected, usable face | 317/320 (99.1%) |
| Calibration query matches | 136/137 correct; zero false matches |
| Held-out query matches | 131/140 correct; nine rejected; zero false matches |
| Held-out negative anchor comparisons | 2,660 |
| Incremental held-out grouping | 160 photos, 20 identities, 23 groups |
| Incremental grouping pair precision / recall | 100% / 95.7% (536/560 same-person pairs recovered) |
| Incremental false merges | 0 |
| Median / p95 bitmap inference latency | 60 / 110 ms |
| Peak sampled process PSS during instrumentation | 203 MiB |

Calibration selects the highest-recall threshold/margin combination with no calibration false matches. Holding that policy fixed, acceptance requires no held-out false matches or incremental false merges, at least 60% held-out query and grouping recall, at least 1,000 negative comparisons, and detection on at least 80% of input photos. The app disables automatic matching if the bundled validated profile is absent or incompatible with its models/pipeline.

These are small-fixture results on aligned public-figure photos, not demographic or personal-album accuracy guarantees. Poses, blur, low light, occlusion, aging, and lookalikes can produce missed or incorrect groups. The reported latency excludes MediaStore decoding, database work, and UI rendering; process PSS includes instrumentation and application overhead. A larger personal-album evaluation remains useful before treating these settings as production validated.

## Reproduce validation

```sh
curl -L -s --fail https://ndownloader.figshare.com/files/5976015 -o artifacts/lfw-funneled.tgz
python3 scripts/prepare-people-validation.py
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb push artifacts/people-validation /data/local/tmp/
adb shell run-as dev.gemmabench mkdir -p files/people-validation
adb shell run-as dev.gemmabench cp -R /data/local/tmp/people-validation/. files/people-validation/
adb shell am instrument -w -e class dev.gemmabench.PeopleRepositoryInstrumentedTest,dev.gemmabench.PeopleFaceInstrumentedTest,dev.gemmabench.PeopleBatchInstrumentedTest -e peopleValidation true dev.gemmabench.test/androidx.test.runner.AndroidJUnitRunner
adb exec-out run-as dev.gemmabench cat files/people-calibration.json
```

Tests cover migration preserving an existing object result, identities across database reopen, manual moves and merges, correction persistence, invalid/ambiguous embeddings, failed or inaccessible anchors, revision invalidation, stop/resume, cache reuse, blank images, native model loading, held-out grouping, and EXIF rotation. The opt-in fixture benchmark is skipped during ordinary instrumented runs. To refresh the bundled policy, review the generated report and copy it to `app/src/main/assets/people/calibration.json` only when `validated` is true. Changing models/preprocessing requires a new calibration and pipeline identity.
