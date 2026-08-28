# ROCm release binaries and Docker images

Date: 2026-08-28

## Goal

Make ROCm a first-class published backend alongside CPU, Vulkan, CUDA, and
Metal. Each tagged release will provide Linux x64 ROCm bundles for the CLI and
server and for the shared C API. The container workflow will publish matching
ROCm CLI and server images. Documentation will describe the supported AMD GPU
targets, host requirements, build flags, artifact names, container tags, and
device passthrough.

The initial runtime and validation target is ROCm 7.2.4 on Ubuntu 24.04. The
hardware gate is the Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) available as
`strix:gpu0` through `rc`.

## Current evidence

Clean commit `f469a57` builds with the pinned ggml v0.13.0 HIP backend when
configured for `gfx1151`. On Strix Halo with ROCm 7.2.4, the 110M F16 TDT model
produced the exact reference transcript.

The warmed transcription measurements for `tests/fixtures/speech.wav` were:

| Backend | Processing time | Transcript |
| --- | ---: | --- |
| ROCm 7.2.4 | 34.505 ms | Exact reference |
| Vulkan / RADV | 53.914 ms | Exact reference |
| CPU, 8 threads | 68.015 ms | Exact reference |

These measurements establish viability and guide documentation. They are not a
performance threshold in CI because runner load and driver versions vary.

## Approaches considered

### One fat ROCm bundle that uses the host ROCm runtime

Build one Linux x64 HIP backend containing code objects for a curated set of
AMD architectures. Package parakeet and ggml, but require a compatible ROCm
userspace installation on the host.

This is the selected approach. It gives users one clearly named artifact and
keeps the release download reasonably sized. It follows the shape of upstream
ggml/llama.cpp ROCm releases.

### Fully self-contained ROCm release bundle

Bundle the HIP runtime, hipBLAS, rocBLAS, rocSOLVER, hipBLASLt, and all
architecture databases. This would make the tarball several gigabytes and
couple it tightly to a driver/runtime combination. The installed ROCm 7.2.4
development stack used for the probe occupied more than 8 GB; hipBLASLt alone
occupied about 4.5 GB. This option is rejected for release tarballs.

### Separate Radeon and Instinct bundles

Split code objects and runtime guidance into consumer/APU and datacenter
artifacts. This reduces each individual HIP library but multiplies assets,
documentation paths, and support ambiguity. It remains a fallback only if the
fat HIP artifact exceeds GitHub artifact or release limits.

## Supported GPU targets

The Linux x64 release and Docker images will build the following HIP targets:

```text
gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201
```

This covers the currently relevant ROCm-supported AMD Instinct generations,
RDNA2/RDNA3/RDNA4 Radeon GPUs, and Ryzen AI APUs including Strix Halo. The list
matches the broad target set used by upstream llama.cpp ROCm containers. It is
passed explicitly through `GPU_TARGETS`; builds must not depend on compiler
auto-detection from the GPU-less GitHub runner.

ROCm images and release artifacts are Linux x86-64 only. CPU and Vulkan remain
the portable choices for AMD hardware outside this target list or on other
operating systems.

## Build configuration

The release and Docker builds use:

```text
-DPARAKEET_GGML_HIP=ON
-DGPU_TARGETS=<supported target list>
-DGGML_HIP_NO_VMM=ON
-DGGML_NATIVE=OFF
```

`GGML_HIP_NO_VMM=ON` is required for predictable behavior on Strix Halo, whose
HIP device reports no VMM support. The existing persistent `ggml_gallocr` fast
path and zero-copy device weight realization remain unchanged.

The HIP compiler and ROCm root are set explicitly when CMake cannot infer them:

```text
-DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++
-DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.4
```

The pinned ggml HIP backend does not support a fully static ggml build. ROCm
artifacts therefore ship the required ggml shared libraries next to the
executables or `libparakeet.so`, with an `$ORIGIN` runtime search path. This is
an implementation detail of the bundle; the public CLI, server, and C API stay
unchanged.

## Release assets

Add one `rocm` / `x64` entry to the Linux release matrix. Use an Ubuntu 24.04
runner and install the ROCm 7.2.4 HIP compiler, device libraries, hipBLAS, and
rocBLAS development packages from AMD's official repository.

Tagged releases and manual workflow runs produce:

```text
parakeet-<version>-bin-linux-rocm-x64.tar.gz
parakeet-<version>-lib-linux-rocm-x64.tar.gz
```

The binary bundle contains:

- `parakeet-cli`
- `parakeet-server`
- the ggml base, CPU, and HIP shared libraries required by the executables
- `LICENSE`
- `README.md`

The library bundle contains:

- `libparakeet.so`
- the ggml base, CPU, and HIP shared libraries required by `libparakeet.so`
- `include/parakeet_capi.h`
- `LICENSE`
- `README.md`

Both bundles require a compatible ROCm 7.2 userspace installation on the host.
Packaging verifies this boundary with `ldd` from outside the build tree. Every
non-system dependency must resolve either from the bundle or from the
documented ROCm runtime.

The existing release upload job needs no new publication mechanism: the ROCm
matrix entry emits the same binary and library artifact outputs as the other
Linux backends.

## Docker images

Extend the existing Docker matrix with a `rocm` variant for Linux `amd64`.
ROCm is not added to the arm64 matrix.

The existing image names gain the following tags:

```text
ghcr.io/mudler/parakeet.cpp-cli:latest-rocm
ghcr.io/mudler/parakeet.cpp-server:latest-rocm
ghcr.io/mudler/parakeet.cpp-cli:<version>-rocm
ghcr.io/mudler/parakeet.cpp-server:<version>-rocm
ghcr.io/mudler/parakeet.cpp-cli:sha-<commit>-rocm
ghcr.io/mudler/parakeet.cpp-server:sha-<commit>-rocm
```

CPU retains the unsuffixed `latest` tag. CUDA retains `latest-cuda`.

The Docker build keeps `ubuntu:24.04` as its build and runtime base and
registers AMD's official ROCm 7.2.4 package repository. The build stage installs
the same minimal development set validated on Strix Halo: `hipcc`, `hip-dev`,
`rocm-device-libs`, `hipblas-dev`, and `rocblas-dev`. The ROCm runtime stage
installs `rocm-hip-runtime` and `rocm-hip-libraries`, then receives the staged
ggml shared libraries. Unlike the thin release tarball, the ROCm container is
turnkey once the host kernel driver exposes the GPU devices.

Users run the image with at least:

```text
--device=/dev/kfd --device=/dev/dri --group-add video
```

Models and audio remain external. The CLI image keeps the `parakeet-cli`
entrypoint, and the server image keeps `parakeet-server --host 0.0.0.0`, exactly
as the CPU and CUDA variants do.

Pull requests continue building only the CPU Docker variants. ROCm joins CUDA
on pushes to `master`, version tags, and manual dispatch because its multi-GPU
code generation and base image are expensive. The merge job creates a
single-platform `linux/amd64` manifest for each ROCm CLI/server tag, using the
same digest and metadata flow as the other variants.

## Validation

### Build-time validation

The release and container workflows must:

1. Configure HIP with the explicit target list and `GGML_HIP_NO_VMM=ON`.
2. Build the CLI and server.
3. Build the shared C API library in a separate tree.
4. Run the existing usage-banner smoke checks.
5. Inspect staged binaries and libraries with `ldd` after moving them outside
   the build tree.
6. Fail if a required ggml library is missing from the package.

GitHub-hosted runners do not need an AMD GPU. They compile code objects for the
explicit architecture list and perform non-device packaging checks.

### Strix Halo hardware gate

Before merging, copy or check out the implementation branch into the shared
`rc` workspace and run all GPU work through `rc run -d strix:gpu0`. Never run
directly on the GPU host without a lease.

Validate the release-style build, extracted bundles, and Docker images:

1. `rocminfo` reports `gfx1151`.
2. parakeet selects `ROCm0`, not CPU or Vulkan.
3. Model-independent tests pass.
4. The extracted CLI bundle transcribes the 110M F16 anchor model and produces
   the exact reference transcript.
5. The extracted server bundle serves an OpenAI-compatible transcription
   request with the same transcript.
6. The extracted shared-library bundle passes a C-API load, transcribe, and
   free smoke test.
7. The CLI ROCm image produces the reference transcript with `/dev/kfd` and
   `/dev/dri` passed through.
8. The server ROCm image returns the reference transcript over HTTP.
9. Warmed ROCm, Vulkan, and CPU timings are recorded for documentation and
   regression context.

The transcript is a correctness gate. Timing is informational unless a later
performance specification introduces a stable threshold.

## Documentation

Update `README.md` to include:

- `rocm` in the Linux x64 release matrix
- the two ROCm release asset names
- the supported HIP target list and representative GPU families
- ROCm 7.2 host runtime requirements and an official installation link
- a source-build example with the HIP CMake flags
- automatic `ROCm0` selection and `PARAKEET_DEVICE=ROCm0` override
- the `latest-rocm`, versioned, and commit Docker tags
- Docker device-passthrough examples for both CLI and server
- the Strix Halo correctness and indicative performance results

Update `examples/server/README.md` with the ROCm server image tag, device
passthrough, and model mounting/fetching examples.

Comments in `.github/workflows/release.yml`, `.github/workflows/docker.yml`,
and `Dockerfile` must describe CPU, CUDA, and ROCm behavior accurately after
the matrix expansion.

## Error handling and compatibility

- If no compiled HIP code object matches the user's GPU, ggml will fail at
  device execution. Documentation directs unsupported GPUs to Vulkan or CPU.
- If ROCm userspace is absent for a release tarball, the dynamic loader error
  is expected; documentation lists the runtime prerequisite.
- Docker users must expose `/dev/kfd` and `/dev/dri`. Documentation calls this
  out next to every ROCm run example.
- `PARAKEET_DEVICE=cpu` continues to force CPU even in a ROCm build.
- The public C and C++ APIs, model format, decoder behavior, and ABI version do
  not change.
- No changes may replace the persistent allocator, introduce per-call weight
  copies, or route supported HIP graphs through the scheduler fast path.

## Out of scope

- Windows HIP/ROCm release artifacts
- ROCm on arm64
- Installing or replacing the host kernel driver
- Bundling a multi-gigabyte ROCm userspace into release tarballs
- Backend-specific kernel optimization or rocWMMA tuning
- Changing decoding, model conversion, quantization, or the public API
