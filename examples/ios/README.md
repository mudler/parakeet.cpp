# ParakeetDemo (iOS)

Live on-device streaming ASR: tap the mic, talk, watch the transcript appear.
Waveform on top, start/stop button in the middle, live transcript at the bottom.

It links `libparakeet` (static, Metal) and drives the streaming C-API
(`parakeet_capi_stream_*`) over mic audio captured with AVAudioEngine and
resampled to 16 kHz mono.

## Build (4 steps)

Requires **full Xcode** (not just Command Line Tools) — the iOS SDK and
`xcodebuild -create-xcframework` need it. If `xcode-select -p` points at
`CommandLineTools`, switch it once:

```sh
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -license accept
```

```sh
cd examples/ios

# 1. Build the static framework for device + simulator (a few minutes).
./scripts/build_xcframework.sh            # -> vendor/Parakeet.xcframework

# 2. Drop a streaming GGUF in as the bundled model (any *streaming* parakeet model).
mkdir -p ParakeetDemo/Resources
cp ~/Downloads/nemotron-3.5-asr-streaming-0.6b-q4_k.NEW.gguf ParakeetDemo/Resources/model.gguf

# 3. Generate the Xcode project.
xcodegen generate                          # -> ParakeetDemo.xcodeproj

# 4. Open and run on a device (or Apple-Silicon simulator).
open ParakeetDemo.xcodeproj
```

Set your signing team in Xcode (Signing & Capabilities) before running on a
physical device.

## Notes / known ceilings

- **Model must be a streaming model** (`nemotron-3.5-asr-streaming-0.6b` or
  `parakeet_realtime_eou_120m-v1`). Offline-only GGUFs will fail `stream_begin`.
- **q4_k is ~0.7 GB** in the app bundle — fine for a dev build; for the App Store
  you'd download it on first launch instead of bundling.
- **Metal** runs on device and on Apple-Silicon simulators. Intel simulators fall
  back to CPU.
- The mic→16 kHz path uses `AVAudioConverter`. Language is chosen in the UI picker
  (`TranscriberModel.lang`, default `en`) and passed to `ParakeetSession.begin(lang:)`;
  `auto` tends to mis-detect, so the picker defaults to a forced locale.
- This is a demo: no error UI beyond a status line, single utterance buffer.
