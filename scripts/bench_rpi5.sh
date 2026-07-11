#!/bin/bash
# Pi 5 benchmark harness for parakeet.cpp: model x threads x clip matrix with
# a thermal/freq/fan/throttle sampler running alongside each timed bench.
# Resumable: completed cells (non-empty results/<tag>.json) are skipped.
#
# Runs ON the Pi, from a directory laid out as:
#   parakeet-cli            static aarch64 build (docker/Dockerfile.static)
#   models/<name>.gguf      the models listed in MODELS below
#   audio/speech.wav        tests/fixtures/speech.wav (7.4 s)
#   audio/speech-60s.wav    the same clip tiled to ~60 s
#
# Pin the cpufreq governor to `performance` first so the numbers measure the
# silicon, not the governor; results land in results/ as
# <model>-t<threads>-<clip>.json plus a .thermal.csv sidecar sampled every 2 s
# (epoch, m°C, kHz, fan rpm, vcgencmd throttled flags). See benchmarks/rpi5.md.
set -eu
cd "$(dirname "$0")"

CLI=${CLI:-./parakeet-cli}
FAN=$(for h in /sys/class/hwmon/hwmon*; do
        [ "$(cat "$h/name")" = pwmfan ] && echo "$h/fan1_input" && break
      done)

mkdir -p results manifests
# Manifests: clip repeated N times -> N timed passes per bench invocation
# (bench loads once, warms up once untimed, then times transcribe only).
for _ in 1 2 3 4 5; do echo "$PWD/audio/speech.wav"; done > manifests/speech.tsv
for _ in 1 2 3;     do echo "$PWD/audio/speech-60s.wav"; done > manifests/speech-60s.tsv

sampler() {
  while :; do
    printf '%s,%s,%s,%s,%s\n' "$(date +%s)" \
      "$(cat /sys/class/thermal/thermal_zone0/temp)" \
      "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)" \
      "$(cat "$FAN")" \
      "$(vcgencmd get_throttled | cut -d= -f2)"
    sleep 2
  done
}

extra_args() {
  case "$1" in
    nemotron-*) echo "--decoder tdt --lang en-US" ;;
    *)          echo "" ;;
  esac
}

MODELS="tdt_ctc-110m-q8_0 realtime_eou_120m-v1-f16 tdt-0.6b-v3-q8_0 nemotron-3.5-asr-streaming-0.6b-q8_0"

for model in $MODELS; do
  for t in 1 2 4; do
    for clip in speech speech-60s; do
      tag="${model}-t${t}-${clip}"
      [ -s "results/$tag.json" ] && { echo "skip $tag (done)"; continue; }
      echo "=== $tag $(date +%H:%M:%S) ==="
      sampler > "results/$tag.thermal.csv" & sp=$!
      # shellcheck disable=SC2046
      $CLI bench --model "models/$model.gguf" --manifest "manifests/$clip.tsv" \
        --threads "$t" $(extra_args "$model") --json "results/$tag.json" \
        > "results/$tag.log" 2>&1 || echo "FAILED: $tag (see results/$tag.log)"
      kill $sp 2>/dev/null || true; wait $sp 2>/dev/null || true
    done
  done
done

# Sanity transcripts, one per model at 4 threads.
for model in $MODELS; do
  # shellcheck disable=SC2046
  $CLI transcribe --model "models/$model.gguf" --input audio/speech.wav \
    --threads 4 $(extra_args "$model") \
    > "results/transcribe-$model.txt" 2>"results/transcribe-$model.err" \
    || echo "TRANSCRIBE-FAILED: $model"
done

echo ALL-DONE
