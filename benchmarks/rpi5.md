# Raspberry Pi 5 (active cooling) benchmark

CPU-only RTFx for the four cached models on a fan-cooled Pi 5, with thermal,
clock, fan and throttle telemetry sampled alongside every run — the point was
both the speed numbers and whether the Active Cooler holds the clocks. It
does: **every cell ran the full 2.4 GHz with `throttled=0x0`**, peaking at
77.1 °C (soft limit 85 °C), fan topping out around 9.7 k rpm on the 0.6b
models from a 2.7 k rpm / 50 °C idle.

Raw per-run JSON and 2-second telemetry CSVs: `results/rpi5/`.

## Setup

| | |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1, 8 GB, Active Cooler |
| OS / kernel | Raspberry Pi OS (Debian 13 trixie), 6.18.34+rpt-rpi-2712, glibc 2.41 |
| CPU | 4× Cortex-A76 @ 2.4 GHz, governor pinned to `performance` for the runs |
| Binary | static aarch64 `parakeet-cli` from `docker/Dockerfile.static` (`GGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16`, in-tree ggml patches applied) |
| Harness | `scripts/bench_rpi5.sh` — `parakeet-cli bench`, median over 5 passes (7.4 s clip) / 3 passes (60 s clip), model load excluded; nemotron runs `--decoder tdt --lang en-US` |
| Audio | `tests/fixtures/speech.wav` (7.4 s) and the same clip tiled to 67 s |

## RTFx (audio seconds per compute second; > 1 = real time)

`speech.wav`, 7.4 s:

| model | t=1 | t=2 | t=4 | load ms |
|---|---|---|---|---|
| tdt_ctc-110m-q8_0 | 6.24 | 11.20 | 18.00 | 52 |
| realtime_eou_120m-v1-f16 | 5.00 | 8.91 | 14.13 | 74 |
| tdt-0.6b-v3-q8_0 | 1.41 | 2.66 | 4.61 | 250 |
| nemotron-3.5-asr-streaming-0.6b-q8_0 (offline) | 1.25 | 2.23 | 3.45 | 255 |

67 s clip (long-form / chunked path):

| model | t=1 | t=2 | t=4 |
|---|---|---|---|
| tdt_ctc-110m-q8_0 | 3.96 | 7.02 | 10.85 |
| realtime_eou_120m-v1-f16 | 3.46 | 6.15 | 9.50 |
| tdt-0.6b-v3-q8_0 | 1.10 | 2.03 | 3.43 |
| nemotron-3.5-asr-streaming-0.6b-q8_0 (offline) | 1.01 | 1.81 | 2.79 |

Takeaways:

- The **110m hybrid is 18× real time** on 4 threads and still 6× on one; the
  small models are the obvious Pi fit.
- The **0.6b models are comfortably real time at 4 threads** (3.4–4.6×) and
  hold ≥ 1× even single-threaded on the long clip. tdt-0.6b-v3-q8_0 peaks at
  ~1.0 GB RSS (4 threads, `VmHWM`), so even a 2 GB Pi is within reach and a
  4 GB one has plenty of headroom.
- Thread scaling is healthy: ~1.8× from 1→2 threads and ~1.6× from 2→4
  across the board.
- Long-form costs roughly a third of the short-clip RTFx (attention window
  growth in the chunked path); scaling with threads is unchanged.
- For scale: the same nemotron q8_0 measurement (same clip, same
  methodology) does 30.8× on a Ryzen 9 9950X3D at 8 threads
  (`BENCHMARK.md`), so the Pi lands at ~1/9th of a big desktop per clip.

## Thermals — what the fan buys

Telemetry per cell in `results/rpi5/*.thermal.csv`
(epoch, m°C, cpu kHz, fan rpm, `vcgencmd get_throttled`):

- **No throttling in any of the 24 cells** (`0x0` everywhere, including the
  sustained back-to-back 0.6b runs); `scaling_cur_freq` never left 2 400 000.
- Peak temperature 77.1 °C (nemotron, 60 s clip); the small models stay
  ≤ 76 °C, short-clip cells ≤ 72.2 °C.
- Fan: 2.7 k rpm at 50 °C idle, ~6.3 k on the 110m, saturating ~9.7 k rpm on
  the 0.6b models.

## Output sanity

All four models transcribe `speech.wav` correctly on the Pi
(`results/rpi5/transcribe-*.txt`): the tdt/hybrid transcripts are
byte-identical to the x86 reference text, `realtime_eou` is lowercase and
unpunctuated by design, and nemotron emits clean text with no `<en-US>`-style
LID tags. Per-clip transcripts are identical across every pass and thread
count.

## Reproducing

```sh
# on the build host
docker buildx build --platform linux/arm64 -f docker/Dockerfile.static \
    --target export --output type=local,dest=build/arm64 .

# on the Pi: dir with parakeet-cli, models/*.gguf, audio/{speech,speech-60s}.wav
sudo sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $g; done'
./bench_rpi5.sh   # scripts/bench_rpi5.sh
```

The aarch64 AppImage (`--target export-appimage`) was validated on the same
Pi: direct FUSE execution, identical transcript.
