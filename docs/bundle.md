# Bundle GGUF

A bundle is one GGUF file that holds several single-model GGUF files, so a user
passes one file instead of several. A bundle can hold six kinds of component:
an ASR model (with its own VAD head, if it has one), a VAD model (Silero, or a
VAD-only slice), speaker diarization, sound-event tagging (ced.cpp), speaker
identification (voice-detect.cpp) and the audio side of a speech LLM.

A bundle is a packaging format. It does not change how any model runs, and it
does not convert or re-quantise any weight: each tensor is copied byte for byte
with its original type. An F16 ASR model can sit next to an F16 Silero model, or
a Q8_0 ASR model next to either, and each keeps the type it was published with.

Single-model GGUF files are unchanged. Every existing loader, converter and
published file works as before.

## Layout

A bundle is a normal GGUF file (version 3) with these rules.

### Header keys

The header keys are the only keys that are not inside a component.

| Key | Type | Meaning |
|---|---|---|
| `general.architecture` | string | `parakeet-bundle` |
| `general.name` | string | Name of the bundle |
| `general.license` | string | Always `other`. The real licences are per component |
| `general.license.name` | string | Always `per-component` |
| `parakeet.bundle.version` | uint32 | Format version. This document describes version 1 |
| `parakeet.bundle.components` | string array | Component names, in file order. At least one. No duplicates |

### Component names

A component name is 1 to 32 characters from `a-z`, `0-9` and `_`, and starts with
a letter. The names `general`, `parakeet` and `bundle` are reserved. A name
never contains a dot, so `<name>.` is an unambiguous prefix.

By convention the components are named after their role: `asr`, `vad`, `diar`,
`ced` and `voice`. The name has no meaning to the loader: the `kind` key does.

The prefixed tensor name must stay within 63 bytes (see "Namespacing"). A few
speaker encoders have long tensor names: the ECAPA GGUF has names of 61 bytes, so
a bundle can only hold it under a one-letter component name (for example `v`).
The build script reports the limit.

### Namespacing

Every tensor and every key of the source file is stored with the prefix
`<component>.`, with no exception and no renaming.

| Source file | Bundle |
|---|---|
| tensor `encoder.layers.0.norm.weight` | tensor `asr.encoder.layers.0.norm.weight` |
| key `parakeet.encoder.d_model` | key `asr.parakeet.encoder.d_model` |
| key `general.architecture` (value `parakeet`) | key `asr.general.architecture` |
| key `general.license` | key `asr.general.license` |
| tensor `vad16k.stft.forward_basis_buffer` | tensor `vad.vad16k.stft.forward_basis_buffer` |
| key `silero_vad.sample_rates` | key `vad.silero_vad.sample_rates` |

The one key that the bundle drops is `general.alignment`, which GGUF defines per
file. The bundle has its own alignment.

A component loader reads the keys and tensors under its prefix and sees them
under their original names. After that it behaves as if it had read the source
file. Tensor names are limited by ggml to 63 bytes, and this limit applies to
the prefixed name. The build script checks it.

A tensor or key outside the header keys and the component prefixes is an error
(`bundle_gguf.py --verify` reports it).

### Per-component keys

Every component has these keys, all strings. All of them except the two hashes
are required.

| Key | Meaning |
|---|---|
| `parakeet.bundle.<c>.kind` | `asr`, `vad`, `diar`, `ced`, `voice` or `speechlm_audio` in version 1 (see "Kinds") |
| `parakeet.bundle.<c>.license` | SPDX identifier, or `LicenseRef-<name>` |
| `parakeet.bundle.<c>.license_url` | `http(s)` URL of the licence text |
| `parakeet.bundle.<c>.source` | Upstream model id or URL |
| `parakeet.bundle.<c>.attribution` | Credit line the licence asks to keep (for example the copyright line) |
| `parakeet.bundle.<c>.changes` | What was changed from the upstream model (CC-BY and similar licences need this) |
| `parakeet.bundle.<c>.source_sha256` | sha256 of the single-model GGUF the component was built from |
| `parakeet.bundle.<c>.content_sha256` | Digest of the component tensors (below) |

The licence keys are the source of truth for the licences of a bundle. A tool
that shows or checks licences reads these keys, not `general.license`, which
cannot hold more than one value.

`content_sha256` is the sha256 of the following byte stream, over the tensors of
the component in file order: the tensor name without the prefix as UTF-8 and a
zero byte, the ggml type id as 4 bytes (little endian), the number of dimensions
as 4 bytes, each dimension as 8 bytes (little endian, in GGUF order), and the
raw tensor data. `bundle_gguf.py --verify` recomputes it, so it detects a
damaged file without the source files.

### Kinds

| Kind | Content | Loaded by |
|---|---|---|
| `asr` | A parakeet ASR GGUF (`general.architecture` `parakeet`, any family that `Model::load` accepts). If it has a VAD head (Ultra, Redux), the head is part of the component | `ModelLoader::load_component`, `Model::load(path, name)` |
| `vad` | A Silero VAD GGUF (`general.architecture` `silero_vad`), or a VAD-only slice of an Ultra or Redux model (`general.architecture` `parakeet`, `parakeet.arch` `vad`, made by `scripts/slice_vad_gguf.py`) | Silero: `SileroVad::load(path, &err, name)`; slice: `Model::load_vad_only(path, name)` |

A `vad` component that holds a slice is told apart from a Silero one by its
`parakeet.arch` key. `parakeet_capi_load_component` gives a VAD-only context for a
slice, the same as `parakeet_capi_load` on the standalone slice file: only the
`parakeet_capi_vad_*` calls work with it. `bundle_gguf.py` accepts a slice only
as kind `vad` and refuses it as kind `asr`. An `asr` component that carries its
own VAD head needs no slice: the head is part of the component. Two cases are
worth knowing:

* `parakeet-cli vad --model bundle.gguf` without `--component` uses the first
  `vad` component, slice or Silero.
* `transcribe --vad` on a bundle looks for a Silero component, else uses the head
  of the ASR component. It does not use a slice component (the ASR component
  already has the head), and `--vad-component` that names a slice is an error.

| `diar` | A Nemotron diarization GGUF (`parakeet.arch` `diarization`) | `DiarizationModel::load(path, name)`, through the same prefixed loader as `asr` |
| `speechlm_audio` | The audio side of a speech LLM (`parakeet.arch` `speechlm_audio`, [speechlm-audio.md](speechlm-audio.md)). It needs a `diar` component beside it | `SpeechLmAudio::load(path, name, path, diar_name)`, `parakeet_capi_load_speechlm_audio` |
| `ced` | A ced.cpp GGUF (`general.architecture` `ced`, CED-tiny, -mini, -small or -base) | `CedTagger::load(path, name)`, from a read-only map of the bundle (see "ced and voice components") |
| `voice` | A voice-detect.cpp speaker encoder GGUF (`general.architecture` `voicedetect` with an embedding: WeSpeaker ResNet34, ECAPA, ERes2Net, CAM++) | `SpeakerEncoder::load(path, name)`, from a read-only map of the bundle |

A reader skips components of a kind it does not know: they are listed, but it does
not load them and does not fail because they exist. The wav2vec2 analysis heads of
voice-detect.cpp (age, gender, emotion) are not a `voice` component: their licence
is non-commercial, and the build script refuses them.

## ced and voice components

`ced.cpp` and `voice-detect.cpp` can load a model from memory:
`ced_capi_load_from_memory_prefixed(data, size, prefix)` and
`voicedetect_capi_load_from_memory_prefixed(data, size, prefix)`. They take the
bytes of a whole GGUF that holds the model under a prefix (`ced.`, `voice.`, the
component name and a dot), parse its header, and copy only the tensors of that
prefix into memory of their own. parakeet.cpp uses them as follows:

1. It reads the bundle header and checks that the component exists and has the
   right kind.
2. It maps the bundle file read-only (`mmap` on Linux and macOS,
   `MapViewOfFile` on Windows; `src/bundle_map.cpp`). A map reads nothing by
   itself: the operating system loads only the pages that are touched.
3. It calls the loader with the map and the prefix. The loader touches the header
   and the tensors of the component, copies them, and returns. The map is
   released at once. The loader keeps no pointer into it.

There is no standalone copy of the component, no temporary file, no memory file
(`memfd`), and no environment variable. The bundle file is opened read-only for
the moment it takes to map it, and closed again. Linux, macOS and Windows run the
same code; only Linux is run in the tests and in CI here.

**Memory and I/O.** The other components are never read. Measured with the
published standard bundle (1100.8 MB: asr 940.5 MB, diar 108.6 MB, ced-small Q8_0
23.6 MB, voice WeSpeaker ResNet34 F32 26.5 MB, vad), loading one component in a
fresh process:

| Component | Component size | Peak resident memory above an idle process | Memory kept after the load | Storage read (page cache dropped before) |
| --- | --- | --- | --- | --- |
| `ced` | 23.6 MB | 49 MB | 25 MB | 24.2 MB |
| `voice` | 26.5 MB | 54 MB | 27 MB | 26.5 MB |

The peak is the copy of the component (anonymous memory, kept) plus the pages of
the map that were touched (file cache, reclaimable, released with the map). It
depends on the size of the component and not on the size of the bundle: the 338 MB
small bundle gives the same figures. The previous design wrote the component to an
in-memory file first, so it held one more copy of the component while the loader ran (not measured).

**Limits.**

* A model file must not change while it loads. If another process shortens the
  bundle while it is mapped, a read past the new end ends the process
  (`SIGBUS`, or an access violation on Windows). Model files are meant to be
  immutable in service.
* The whole bundle needs address space, not memory. A 32-bit process cannot map a
  bundle larger than its address space; the load then fails with "cannot map".
* A tensor name in the bundle is `<component>.<name>`, and ggml keeps names in 63
  bytes, so the stripped names the loaders see are shorter than that.

The component loaded this way gives the same output as the single-model file:
the CED class scores and the speaker embeddings are bitwise equal in the tests.
Load errors come from `ced_capi_last_error(NULL)` and
`voicedetect_capi_last_load_error()` and are added to the message of
`CedTagger::load` and `SpeakerEncoder::load`.

The speaker identity of a `voice` component (`parakeet_capi_speaker_identity`) is
`sha256:` plus the `source_sha256` of its header, which is the sha256 of the
single-model file. A voice enrolled with the standalone file therefore matches the
same model inside a bundle. The header is trusted for this: load bundles you
trust.

A `voice` component also embeds audio. `parakeet_capi_speaker_embed_pcm` on a
context from `parakeet_capi_load_component(path, "voice")` returns the embedding
of a clip (see "Embeddings out of the library" in [speaker.md](speaker.md)). The
embeddings are in the same space as the standalone WeSpeaker encoder file the
component was built from: in the test, the cosine between the two is 1.000000
for the 26.5 MB WeSpeaker ResNet34 component of `parakeet-bundle-small.gguf`.
A host can store embeddings from either and compare them.

## Compatibility rules

1. **Plain files are unchanged.** A single-model GGUF has no bundle keys. Every
   loader opens it as before. Passing a component name for a plain file is an
   error.
2. **Old readers refuse a bundle.** A bundle has no top-level `parakeet.*` model
   keys (no `parakeet.encoder.d_model`, no `parakeet.vocab_size`) and no
   `silero_vad.*` keys, and `general.architecture` is `parakeet-bundle`, which no
   earlier loader accepts. A reader from before this format fails on a missing
   key and reports a load failure. It never reads the wrong weights. Tools that
   only list the GGUF header still work and show the bundle keys.
3. **New readers refuse a plain call on a bundle with a clear message** instead
   of failing on a missing key: `ModelLoader::load` and `SileroVad::load`
   without a component say that the file is a bundle and that a component must
   be named.
4. **Version.** `parakeet.bundle.version` is 1. A reader refuses a bundle with a
   higher version and says which version it knows. A future change that keeps
   old readers working (new optional keys, new kinds) does not change the
   version. A change that would make a version 1 reader misread a file does.
5. **Unknown keys and kinds are ignored** by a reader. A bundle may carry extra
   `parakeet.bundle.<c>.*` keys.
6. **Header is read alone.** Listing the components, licences and sizes reads the
   header and the tensor table only, never tensor data.

## Selection rules

A component is chosen by name or by default.

* **Default component** (`parakeet_capi_load`, `transcribe` without
  `--component`): the only component of kind `asr`. A bundle with no `asr`
  component but exactly one loadable component of another kind opens that one
  (a Silero-only bundle opens as a VAD). If more than one candidate exists the
  load fails and the message lists the component names. There is no silent
  choice.
* **By name** (`parakeet_capi_load_component`, `--component`): that component,
  whatever its kind. The context has the kind the component declares.
* **Other kinds** (`diar`, `ced`, `voice`) are never the default when an `asr`
  component exists: `parakeet_capi_load` opens the ASR component, and the others
  are opened by name with `parakeet_capi_load_component`. A bundle without an
  `asr` component but with exactly one loadable component opens that one; with
  several it is refused with a message that lists them.
* **Commands that need one kind** (`parakeet-cli scene`, `enroll`, `info`, the
  `diarize` tool): given a bundle, they use the only component of the kind they
  need (ASR, diarization, sound or speaker) and refuse when there is none or
  several; `--component` (`enroll`, `diarize`) or `--asr-component`,
  `--diar-component`, `--sound-component`, `--speakers-component` (`scene`) name
  one. Naming a component of the wrong kind is an error.
* **VAD source** for `parakeet-cli vad` and for segmented transcription
  (`transcribe --vad`) on a bundle: the Silero component if the bundle has one,
  otherwise the VAD head of the ASR component. `--component` (for `vad`) and
  `--vad-component` (for `transcribe`) override the choice; naming the ASR
  component selects its head. A Silero model passed with `--vad-model` takes
  precedence over the bundle.

## Partial loading

A loader reads the header and the tensor table, then reads only the tensors of
the component it was asked for. The ASR and diarization loaders read them into
one private memory block, and the Silero loader seeks to each of its tensors. The
`ced` and `voice` loaders get a read-only map of the bundle and copy the tensors
of their component out of it (see "ced and voice components"). The bytes of the
other components are never read. A test checks the bytes read from storage for
each component.

Memory: a loaded component uses the same memory as the same single-model file.
The memory of components that are not loaded is not used. For `ced` and `voice`
the map exists only during the load.

## Building, inspecting and verifying

`scripts/bundle_gguf.py` (Python with `gguf` and `numpy`, as for the converter):

```
bundle_gguf.py --out bundle.gguf --manifest manifest.json
bundle_gguf.py --list bundle.gguf [--json]
bundle_gguf.py --verify bundle.gguf [--source asr=asr.gguf --source vad=silero.gguf]
bundle_gguf.py --notice bundle.gguf     # credits and the full licence texts
```

The manifest is a JSON file. `file` is relative to the manifest.

```json
{"name": "parakeet-bundle-standard",
 "components": [
   {"name": "asr", "file": "tdt-0.6b-v3-q8_0.gguf", "kind": "asr",
    "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "source": "nvidia/parakeet-tdt-0.6b-v3",
    "attribution": "Parakeet TDT 0.6B v3 by NVIDIA, licensed CC-BY-4.0",
    "changes": "Converted from the NeMo checkpoint to GGUF; weights quantised to Q8_0"},
   {"name": "vad", "file": "silero-vad-f16.gguf", "kind": "vad",
    "license": "MIT", "license_url": "https://github.com/snakers4/silero-vad/blob/master/LICENSE",
    "source": "https://github.com/snakers4/silero-vad",
    "attribution": "Copyright (c) 2020-present Silero Team",
    "changes": "Converted from ONNX to GGUF (F16)"}]}
```

The build:

* checks every input: it is a GGUF, it is not a bundle, its architecture matches
  the kind, the keys the loader needs exist, tensor names are unique and short
  enough;
* refuses a component with a missing or empty `license`, `license_url`,
  `source`, `attribution` or `changes`;
* refuses a licence that is not an SPDX-style identifier, and refuses licences
  that restrict commercial use or derivatives (`-NC-`, `-ND-`);
* refuses models that must not be bundled: the audeering wav2vec2 heads
  (CC-BY-NC-SA-4.0) and the end-of-utterance ASR model (NVIDIA Open Model
  License: notice duty and a revocation clause), by source, licence or input name;
* refuses a known source with another licence than the table `KNOWN_SOURCES` in
  the script (see "Licences");
* refuses a component whose input file declares a `general.license` that
  differs from the manifest (the same licence in another spelling is accepted);
* records the sha256 of each input file and the content digest of each
  component;
* is deterministic: the same inputs and manifest give the same file, byte for
  byte (no timestamps, keys and tensors in source order);
* writes to a temporary file and renames it, and never overwrites an input.

`--verify` checks the structure (no tensor or key outside a component, valid
names, complete licence keys, tensors inside the file), recomputes every
`content_sha256`, and with `--source` compares the tensors (name, type, shape,
bytes) and the keys with the single-model file and its recorded sha256. It
exits 1 on any problem.

`--notice` prints a NOTICE text from the licence keys: for each component the
model, licence and URL, credit line, changes and input hash, followed by the full
text of every licence used (from `scripts/bundle_licenses/`). It fails when a
licence has no text there. Ship this text with the file.

## Licences

Each component keeps the licence of the model it was converted from. A bundle
is a collection, so it does not have one licence. Rules the tools apply:

* every component states its licence, source, credit and changes in the header;
* `general.license` is `other`, and a tool must read the per-component keys;
* components under licences that restrict commercial use or derivatives are not
  bundled;
* the NOTICE text and the licence texts must be distributed with the file, as
  the licences require (CC-BY-4.0 asks for credit, a licence link and a note of
  changes; MIT asks that the copyright notice and licence text stay with copies).

### Licences of the supported components

| Component | Source | Licence | What the bundle must carry |
|---|---|---|---|
| ASR | NVIDIA Parakeet (NeMo) and Moondream Ultra/Redux | CC-BY-4.0 | credit, licence link, note of changes |
| diar | `nvidia/Nemotron-3-Diarization` | OpenMDW-1.1 | a copy of the agreement and the origin notices |
| ced | `mispeech/ced-*` | Apache-2.0 | licence text and copyright notices |
| voice, ECAPA | `speechbrain/spkrec-ecapa-voxceleb` | Apache-2.0 | licence text and notices |
| voice, ERes2Net | `iic/speech_eres2net_base_200k_sv_zh-cn_16k-common` (3D-Speaker, ModelScope) | Apache-2.0 | licence text and notices |
| voice, WeSpeaker ResNet34 | `Wespeaker/wespeaker-voxceleb-resnet34-LM` | CC-BY-4.0 (see below) | credit to the WeSpeaker project, licence link, note of changes |
| vad | `snakers4/silero-vad` | MIT | copyright notice and licence text |

WeSpeaker: the voice-detect.cpp file is converted from `voxceleb_resnet34_LM.onnx` of
`Wespeaker/wespeaker-voxceleb-resnet34-LM`, and that card says CC-BY-4.0. The card of the
plain `Wespeaker/wespeaker-voxceleb-resnet34` says Apache-2.0, but the WeSpeaker project
states in its `docs/pretrained.md` that its VoxCeleb-trained models follow CC-BY-4.0.
The bundle uses CC-BY-4.0 and credits the WeSpeaker project.

CED: the `mispeech/ced-*` cards say Apache-2.0, and the bundle follows them. The upstream
GitHub code repository is GPL-3.0, and the original checkpoint records on Zenodo say
CC-BY-4.0. The licence of the weights is therefore not consistent upstream. Confirm it with
the authors before a public release. The CC-BY credit to Heinrich Dinkel and Xiaomi
(mispeech) is kept in the meantime.

Not bundled: the audeering heads (CC-BY-NC-SA-4.0) and the EOU model.

### Training data

The speaker models ECAPA and WeSpeaker are trained on VoxCeleb. They carry the terms that
their authors state for the model (WeSpeaker: CC-BY-4.0). Whether a trained model is
derived from its training data is a legal question that this project does not settle.

### Credit strings

Suggested `source`, `attribution` and `changes` values for the voice encoders:

| Model | Source | Credit | Changes |
|---|---|---|---|
| WeSpeaker ResNet34-LM | `https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM` | WeSpeaker ResNet34-LM (large-margin fine-tuned, trained on VoxCeleb2 dev) by the WeSpeaker project (https://github.com/wenet-e2e/wespeaker), licensed under CC-BY-4.0 | Converted from the ONNX export to GGUF (voice-detect.cpp, F32); no other change |
| ECAPA-TDNN | `https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb` | ECAPA-TDNN speaker encoder by the SpeechBrain authors (speechbrain/spkrec-ecapa-voxceleb), trained on VoxCeleb 1 and 2, licensed under Apache-2.0 | Converted from the SpeechBrain embedding model checkpoint to GGUF (voice-detect.cpp); no other change |
| ERes2Net | `https://www.modelscope.cn/models/iic/speech_eres2net_base_200k_sv_zh-cn_16k-common` | ERes2Net-base (200k speakers, zh-cn) by the 3D-Speaker authors (https://github.com/modelscope/3D-Speaker), licensed under Apache-2.0 | Converted from the sherpa-onnx ONNX export to GGUF (voice-detect.cpp, F32); no other change |

## C-API and CLI

```c
parakeet_ctx* parakeet_capi_load(const char* path);                 // bundle: default ASR component
parakeet_ctx* parakeet_capi_load_component(const char* path, const char* name);
// kind asr -> ASR context, diar -> diarization, ced -> sound tagger, voice -> speaker
// encoder, vad -> Silero (or a VAD-only slice). See parakeet_capi_model_kind.
char*         parakeet_capi_bundle_components_json(const char* path);  // free with parakeet_capi_free_string
const char*   parakeet_capi_load_error(void);                          // reason of the last failed load on this thread
```

```
parakeet-cli info bundle.gguf [--component NAME]
parakeet-cli transcribe --model bundle.gguf --input a.wav [--component NAME]
parakeet-cli transcribe --model bundle.gguf --input a.wav --vad [--vad-component NAME]
parakeet-cli vad --model bundle.gguf --input a.wav [--component NAME]
parakeet-cli scene --model bundle.gguf --diar bundle.gguf --sound bundle.gguf --input a.wav
parakeet-cli scene ... --speakers bundle.gguf --registry reg.bin
parakeet-cli enroll --model bundle.gguf [--component NAME] --name N --input a.wav --registry reg.bin
diarize bundle.gguf a.wav [--stream] [--component NAME]
```

Every function that takes a context works with the contexts of a bundle: the
diarization, sound and speaker contexts from `parakeet_capi_load_component`
feed `parakeet_capi_diarize_*`, `parakeet_capi_transcribe_and_diarize_*`, the
scene stream and the named-diarization calls the same way as contexts from
single-model files.

`parakeet-cli info` without `--component` lists the components, kinds, licences,
sources, credits and sizes from the header. `--component` prints the details of
one component. `bench` and the streaming ASR modes do not take a bundle yet: they
refuse it with a message.

### A VAD-only slice in a bundle

A `vad` component can also hold a VAD-only slice of an Ultra or Redux model
(`scripts/slice_vad_gguf.py`, `parakeet.arch` `vad`). It is told apart from a
Silero model by that key. `parakeet_capi_load_component` gives a VAD-only context
(only the `parakeet_capi_vad_*` calls work), the same as `parakeet_capi_load` on the
standalone slice file. The build accepts a slice only as kind `vad`. An `asr`
component with its own VAD head needs no slice. `parakeet-cli vad` uses the first
`vad` component by default; `transcribe --vad` looks for a Silero component, else
uses the head of the ASR component, and does not use a slice.

## Not verified

* GPU backends (the loaders and the compute path are the same as for
  single-model files, but no bundle was run on a GPU);
* cache-aware streaming ASR from a bundle component, and the other ASR families
  (EOU, CTC, Nemotron ASR, K-quants) through a bundle;
* macOS and Windows: the temporary-file fallback for `ced` and `voice` is written
  for them but not run there;
* a bundle with a K-quant or Q4 component.
