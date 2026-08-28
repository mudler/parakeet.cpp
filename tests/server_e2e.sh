#!/usr/bin/env bash
# End-to-end test for parakeet-server. Opt-in: needs network + curl, downloads
# the ~125 MB tdt_ctc-110m-q4_k model via the alias path. Skips (77) unless
# PARAKEET_SERVER_E2E=1.
#
# Usage: server_e2e.sh <path-to-parakeet-server>
# Run from the project root (ctest sets WORKING_DIRECTORY).
set -u

if [ "${PARAKEET_SERVER_E2E:-}" != "1" ]; then
    echo "server_e2e: PARAKEET_SERVER_E2E != 1; skip"
    exit 77
fi

SERVER="${1:?need path to parakeet-server}"
FIXTURE="tests/fixtures/speech.wav"
PORT="${PARAKEET_SERVER_E2E_PORT:-18080}"
READY_TIMEOUT="${PARAKEET_SERVER_E2E_READY_TIMEOUT:-180}"
CACHE="$(mktemp -d)"
EXPECT="turning away her"

# Arm cleanup before the pre-flight guards so a failed guard still removes CACHE.
cleanup() { [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null; rm -rf "$CACHE"; }
trap cleanup EXIT

command -v curl >/dev/null 2>&1 || { echo "server_e2e: curl required"; exit 1; }
[ -f "$FIXTURE" ] || { echo "server_e2e: missing $FIXTURE"; exit 1; }
case "$READY_TIMEOUT" in
    ''|*[!0-9]*) echo "server_e2e: PARAKEET_SERVER_E2E_READY_TIMEOUT must be seconds"; exit 1 ;;
esac
[ "$READY_TIMEOUT" -gt 0 ] || { echo "server_e2e: PARAKEET_SERVER_E2E_READY_TIMEOUT must be > 0"; exit 1; }

"$SERVER" --model tdt_ctc-110m-q4_k --port "$PORT" --cache-dir "$CACHE" &
SRV=$!

# Wait for /health. The server binds only after first-run model download + load.
ready=0
for _ in $(seq 1 "$READY_TIMEOUT"); do
    if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$SRV" 2>/dev/null || { echo "server_e2e: server exited early"; exit 1; }
    sleep 1
done
[ "$ready" = "1" ] || { echo "server_e2e: server not ready after ${READY_TIMEOUT}s"; exit 1; }

base="http://127.0.0.1:$PORT/v1/audio/transcriptions"
fail=0

# json (default)
body=$(curl -fs -F file=@"$FIXTURE" "$base")
echo "json: $body"
echo "$body" | grep -q "$EXPECT" || { echo "FAIL json text"; fail=1; }
echo "$body" | grep -q '"text"' || { echo "FAIL json shape"; fail=1; }

# text
body=$(curl -fs -F file=@"$FIXTURE" -F response_format=text "$base")
echo "text: $body"
echo "$body" | grep -q "$EXPECT" || { echo "FAIL text"; fail=1; }

# verbose_json with word timestamps
body=$(curl -fs -F file=@"$FIXTURE" -F response_format=verbose_json \
            -F 'timestamp_granularities[]=word' "$base")
echo "verbose: $body"
echo "$body" | grep -q '"segments"' || { echo "FAIL verbose segments"; fail=1; }
echo "$body" | grep -q '"words"'    || { echo "FAIL verbose words"; fail=1; }
echo "$body" | grep -q "$EXPECT"    || { echo "FAIL verbose text"; fail=1; }

# 400: missing file (use -o/-w to capture status without -f aborting)
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$base")
[ "$code" = "400" ] || { echo "FAIL missing-file expected 400 got $code"; fail=1; }

# 400: non-WAV upload
code=$(curl -s -o /dev/null -w '%{http_code}' -F file=@"$SERVER" "$base")
[ "$code" = "400" ] || { echo "FAIL non-wav expected 400 got $code"; fail=1; }

# 400: unknown response_format
code=$(curl -s -o /dev/null -w '%{http_code}' \
            -F file=@"$FIXTURE" -F response_format=bogus "$base")
[ "$code" = "400" ] || { echo "FAIL bad-format expected 400 got $code"; fail=1; }

[ "$fail" = "0" ] && { echo "server_e2e: OK"; exit 0; }
echo "server_e2e: FAILED"; exit 1
