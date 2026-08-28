# parakeet-server

A small OpenAI-drop-in HTTP server for transcription, built on parakeet.cpp.
Point any OpenAI client's `base_url` at it and call
`POST /v1/audio/transcriptions`.

This is an example, not a production service. It serves one model, runs one
transcription at a time, and accepts WAV uploads only.

For a production deployment, use [LocalAI](https://localai.io), which embeds
parakeet.cpp as a backend and adds the things this example deliberately leaves
out: a model gallery, concurrency, multi-model serving, the full OpenAI API
surface, auth, and metrics.

## Build

Built by default with the rest of the project (`PARAKEET_BUILD_SERVER=ON`):

```sh
cmake -B build && cmake --build build --target parakeet-server -j
```

## Run

With a local model:

```sh
./build/examples/server/parakeet-server --model path/to/model.gguf --port 8080
```

With a published model by alias (downloaded once and cached under
`${XDG_CACHE_HOME:-$HOME/.cache}/parakeet.cpp/models`, override with
`--cache-dir` or `PARAKEET_CACHE_DIR`):

```sh
./build/examples/server/parakeet-server --model tdt_ctc-110m-q4_k
```

`--model` accepts a local `.gguf` path, an `http(s)://` URL, a `<name>.gguf`
filename in the `mudler/parakeet-cpp-gguf` repo, or one of these aliases:

| Alias                | Model                                   |
|----------------------|-----------------------------------------|
| `tdt_ctc-110m`       | hybrid TDT+CTC 110M (f16)               |
| `tdt_ctc-110m-q4_k`  | hybrid TDT+CTC 110M (q4_k, smallest)    |
| `tdt_ctc-1.1b`       | hybrid TDT+CTC 1.1B (f16)               |
| `tdt-0.6b-v2`        | TDT 0.6B v2 (f16)                       |
| `tdt-0.6b-v3`        | TDT 0.6B v3, multilingual (f16)         |
| `tdt-1.1b`           | TDT 1.1B (f16)                          |
| `ctc-0.6b`           | CTC 0.6B (f16)                          |
| `ctc-1.1b`           | CTC 1.1B (f16)                          |
| `rnnt-0.6b`          | RNN-T 0.6B (f16)                        |
| `rnnt-1.1b`          | RNN-T 1.1B (f16)                        |
| `eou-120m`           | realtime EOU 120M (f16)                 |

Downloads use `curl` (or `wget`). If neither is on `PATH`, download the `.gguf`
yourself and pass the local path.

## Docker

A prebuilt image is published per push to `ghcr.io/<owner>/parakeet.cpp-server`
with CPU, CUDA 13, CUDA 12, and ROCm variants. It binds `0.0.0.0` and exposes
port 8080. CUDA 12 tags use the `-cuda12` suffix. The ROCm variant is Linux
amd64 only. It uses `latest-rocm`, `<version>-rocm`, and `<sha>-rocm` tags. The
`<sha>` value has the `sha-<short-commit>` form.

Pass the same `--model` argument that you use with the local server. You can
let the server fetch an alias or mount a local `.gguf` file:

```sh
# serve a published model by alias (downloaded into the container)
docker run --rm -p 8080:8080 ghcr.io/mudler/parakeet.cpp-server --model tdt_ctc-110m

# serve a local model (mount it read-only)
docker run --rm -p 8080:8080 -v "$PWD/model.gguf:/model.gguf:ro" \
  ghcr.io/mudler/parakeet.cpp-server --model /model.gguf

# serve a published model with ROCm
docker run --rm -p 8080:8080 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  ghcr.io/mudler/parakeet.cpp-server:latest-rocm \
  --model tdt_ctc-110m

# serve a local model with ROCm
docker run --rm -p 8080:8080 \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$PWD/model.gguf:/model.gguf:ro" \
  ghcr.io/mudler/parakeet.cpp-server:latest-rocm \
  --model /model.gguf
```

The ROCm image contains ROCm 7.2.4 userspace. The host supplies its AMD kernel
driver and access to `/dev/kfd` and `/dev/dri`. The user must belong to the
group passed through with `--group-add video`.

The server auto-selects `ROCm0` in the ROCm image. Add
`-e PARAKEET_DEVICE=ROCm0` to select it explicitly. Add
`-e PARAKEET_DEVICE=cpu` to force CPU.

On a Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`, Strix Halo), a warmed 110M
F16 TDT run took 34.505 ms on ROCm. The same run took 53.914 ms on Vulkan and
68.015 ms on CPU with 8 threads. Each backend produced this exact transcript:

> Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. It is certainly very like the old portrait.

These measurements are indicative, not guaranteed. Host load, drivers, and
build options affect processing time.

## Call it

```sh
curl -F file=@audio.wav -F response_format=verbose_json \
  http://localhost:8080/v1/audio/transcriptions
```

With the OpenAI Python client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="parakeet", file=f).text)
```

## Supported

- `response_format`: `json` (default), `text`, `verbose_json`.
- `timestamp_granularities[]=word` adds a `words` array to `verbose_json`.

## Known simplifications

- WAV uploads only. Other formats return 400. Convert with ffmpeg first.
- `verbose_json` emits a single `segment` spanning the whole transcript;
  Parakeet has no native segmentation. Word timestamps are real.
- `language` in `verbose_json` is a fixed `en` placeholder.
- `model` in the request is accepted but ignored; the process serves the one
  model given to `--model`.
- `temperature` and `prompt` are accepted and ignored (greedy decode).
- Inference is serialized by a mutex. For real parallelism, hold a pool of
  `pk::Model` contexts instead of one.
