#!/bin/sh
# Bundle a static parakeet-cli into a binary-only AppImage.
#
#   make-appimage.sh <parakeet-cli> <aarch64|x86_64> <version> <outdir>
#     -> <outdir>/parakeet-cli-<version>-<arch>.AppImage
#
# Single source of truth for AppImage packaging: docker/Dockerfile.static's
# appimage stage and release.yml's cpu matrix entries both call this. The
# binary must already be self-contained apart from libc/libstdc++/libgomp
# (BUILD_SHARED_LIBS=OFF); libstdc++/libgomp are bundled from the running
# system, so this must run on a system whose native arch matches <arch> --
# ldd has to resolve them.
#
# appimagetool (host arch) and the type2 runtime (target arch) are fetched
# from their "continuous" releases unless APPIMAGETOOL / APPIMAGE_RUNTIME
# point at existing files. APPIMAGE_EXTRACT_AND_RUN keeps FUSE out of the
# loop, so this works in containers and on CI runners.
set -eu

BIN=$1
ARCH=$2
VERSION=$3
OUTDIR=$4

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

fetch() {
    if command -v curl >/dev/null; then curl -fsSL -o "$2" "$1"
    else wget -qO "$2" "$1"; fi
}

# --- AppDir ------------------------------------------------------------------
APPDIR=$WORK/AppDir
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/lib"
cp "$BIN" "$APPDIR/usr/bin/parakeet-cli"
chmod +x "$APPDIR/usr/bin/parakeet-cli"

# The two non-libc dynamic deps, resolved from the running system (cp -L
# follows the .so.N symlinks so the AppDir holds real files under the SONAME).
for lib in libstdc++.so.6 libgomp.so.1; do
    path=$(ldd "$BIN" | awk -v l="$lib" '$1 == l { print $3 }')
    if [ -z "$path" ]; then
        echo "error: $lib not found in 'ldd $BIN' output" >&2
        exit 1
    fi
    cp -L "$path" "$APPDIR/usr/lib/$lib"
done

cp "$HERE/AppRun" "$APPDIR/AppRun"
chmod +x "$APPDIR/AppRun"
cp "$HERE/parakeet-cli.desktop" "$APPDIR/parakeet-cli.desktop"
cp "$HERE/parakeet-cli.png" "$APPDIR/parakeet-cli.png"
ln -sf parakeet-cli.png "$APPDIR/.DirIcon"

# --- appimagetool + runtime ---------------------------------------------------
if [ -z "${APPIMAGETOOL:-}" ]; then
    APPIMAGETOOL=$WORK/appimagetool
    fetch "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$(uname -m).AppImage" \
        "$APPIMAGETOOL"
    chmod +x "$APPIMAGETOOL"
    # qemu-user's binfmt mask requires the ELF e_ident pad bytes to be zero,
    # but an AppImage carries 'AI\x02' at offset 8 -- under emulation (the
    # docker/Dockerfile.static arm64 stage) the tool gets "Exec format error"
    # before it even starts. Zero the three magic bytes: plain padding to the
    # ELF loader, and with APPIMAGE_EXTRACT_AND_RUN appimagetool never needs
    # its own AppImage magic.
    printf '\0\0\0' | dd of="$APPIMAGETOOL" bs=1 seek=8 count=3 conv=notrunc 2>/dev/null
fi
if [ -z "${APPIMAGE_RUNTIME:-}" ]; then
    APPIMAGE_RUNTIME=$WORK/runtime-$ARCH
    fetch "https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-$ARCH" \
        "$APPIMAGE_RUNTIME"
fi

# --- pack ---------------------------------------------------------------------
mkdir -p "$OUTDIR"
OUT=$OUTDIR/parakeet-cli-$VERSION-$ARCH.AppImage
ARCH=$ARCH APPIMAGE_EXTRACT_AND_RUN=1 \
    "$APPIMAGETOOL" --runtime-file "$APPIMAGE_RUNTIME" "$APPDIR" "$OUT"
echo "built $OUT"
