#!/usr/bin/env bash
# Build vendor/Parakeet.xcframework (static libparakeet + ggml, Metal embedded)
# for iOS device + simulator, and vendor the C-API header.
#
# Run from examples/ios/:  ./scripts/build_xcframework.sh
# Needs Xcode + cmake. ~a few minutes.
set -euo pipefail

# Point the toolchain at full Xcode for THIS process only (no sudo, no
# machine-wide xcode-select change). Override by exporting DEVELOPER_DIR first.
: "${DEVELOPER_DIR:=/Applications/Xcode.app/Contents/Developer}"
export DEVELOPER_DIR

HERE="$(cd "$(dirname "$0")/.." && pwd)"   # examples/ios
ROOT="$(cd "$HERE/../.." && pwd)"          # repo root
OUT="$HERE/vendor"
rm -rf "$OUT"; mkdir -p "$OUT/include"
cp "$ROOT/include/parakeet_capi.h" "$OUT/include/"

# Common cmake flags: static everything, Metal with the shader source embedded
# (no .metallib to ship), no host-ISA tuning (we're cross-compiling).
COMMON=(
  -G Xcode
  -DCMAKE_SYSTEM_NAME=iOS
  -DPARAKEET_BUILD_CLI=OFF -DPARAKEET_BUILD_SERVER=OFF -DPARAKEET_BUILD_TESTS=OFF
  -DPARAKEET_SHARED=OFF
  -DPARAKEET_GGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
  -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
)

build_slice () {           # $1 = tag, $2 = sysroot, $3 = archs
  local tag="$1" sysroot="$2" archs="$3"
  local b="$HERE/build-$tag"
  rm -rf "$b"
  # Build chatter -> stderr, so this function's stdout is ONLY the merged path
  # (it's captured via $(...) by the caller).
  cmake -S "$ROOT" -B "$b" "${COMMON[@]}" \
    -DCMAKE_OSX_SYSROOT="$sysroot" -DCMAKE_OSX_ARCHITECTURES="$archs" >&2
  cmake --build "$b" --config Release -j >&2
  # Merge every static lib cmake produced into one fat-by-arch archive.
  local merged="$HERE/$tag.a"
  # Only the final per-config libs; prune Xcode's intermediate per-arch archives
  # under build/.../Objects-normal (merging both would duplicate symbols).
  libtool -static -o "$merged" \
    $(find "$b" -path '*/build/*' -prune -o -name '*.a' -print | tr '\n' ' ') >&2 2>&1
  echo "$merged"
}

DEV="$(build_slice device iphoneos arm64)"
SIM="$(build_slice sim iphonesimulator 'arm64;x86_64')"

rm -rf "$OUT/Parakeet.xcframework"
xcodebuild -create-xcframework \
  -library "$DEV" -headers "$OUT/include" \
  -library "$SIM" -headers "$OUT/include" \
  -output "$OUT/Parakeet.xcframework"

echo "OK -> $OUT/Parakeet.xcframework"
