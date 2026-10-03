# Bundle GGUF

A bundle is one GGUF file that holds several single-model GGUF files, so a user
passes one file instead of several. Phase 1 supports two component kinds: an
ASR model (with its own VAD head, if it has one) and a Silero VAD model.

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

By convention the ASR component is named `asr` and the Silero component is
named `vad`. The name has no meaning to the loader: the `kind` key does.

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
| `parakeet.bundle.<c>.kind` | `asr` or `vad` in version 1 (see "Kinds") |
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
| `vad` | A Silero VAD GGUF (`general.architecture` `silero_vad`) | `SileroVad::load(path, &err, name)` |

The names `diarization`, `sound` and `voice` are reserved for later phases. A
reader skips components of a kind it does not know: they are listed, but it does
not load them and does not fail because they exist.

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
* **VAD source** for `parakeet-cli vad` and for segmented transcription
  (`transcribe --vad`) on a bundle: the Silero component if the bundle has one,
  otherwise the VAD head of the ASR component. `--component` (for `vad`) and
  `--vad-component` (for `transcribe`) override the choice; naming the ASR
  component selects its head. A Silero model passed with `--vad-model` takes
  precedence over the bundle.

## Partial loading

A loader reads the header and the tensor table, then reads only the tensors of
the component it was asked for. The ASR loader reads them into one private
memory block, and the Silero loader seeks to each of its tensors. The bytes of
the other components are never read. A test checks the number of bytes read from
the file for each component.

Memory: a loaded component uses the same memory as the same single-model file.
The memory of components that are not loaded is not used. The file stays on disk
and is not memory mapped.

## Building, inspecting and verifying

`scripts/bundle_gguf.py` (Python with `gguf` and `numpy`, as for the converter):

```
bundle_gguf.py --out bundle.gguf --manifest manifest.json
bundle_gguf.py --list bundle.gguf [--json]
bundle_gguf.py --verify bundle.gguf [--source asr=asr.gguf --source vad=silero.gguf]
bundle_gguf.py --notice bundle.gguf
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
model, licence and URL, credit line, changes and input hash. Ship this text and
the licence texts with the file.

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

## C-API and CLI

```c
parakeet_ctx* parakeet_capi_load(const char* path);                 // bundle: default ASR component
parakeet_ctx* parakeet_capi_load_component(const char* path, const char* name);
char*         parakeet_capi_bundle_components_json(const char* path);  // free with parakeet_capi_free_string
const char*   parakeet_capi_load_error(void);                          // reason of the last failed load on this thread
```

```
parakeet-cli info bundle.gguf [--component NAME]
parakeet-cli transcribe --model bundle.gguf --input a.wav [--component NAME]
parakeet-cli transcribe --model bundle.gguf --input a.wav --vad [--vad-component NAME]
parakeet-cli vad --model bundle.gguf --input a.wav [--component NAME]
```

`parakeet-cli info` without `--component` lists the components, kinds, licences,
sources, credits and sizes from the header. `--component` prints the details of
one component. Other subcommands (bench, scene, streaming) do not take a bundle
yet: they refuse it with a message.

## Not covered in phase 1

* diarization, sound-event and speaker-identification components (the kinds are
  reserved; their loaders are separate libraries);
* cache-aware streaming from a bundle component;
* GPU backends were not run with bundles (the loader and the compute path are
  the same as for single-model files, but there is no test).
