#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 SOURCE_ROOT DESTINATION" >&2
    exit 2
fi

source_root=$1
destination=$2

if [[ ! -d "$source_root" ]]; then
    echo "source root is not a directory: $source_root" >&2
    exit 2
fi

if [[ ! -d "$destination" ]]; then
    echo "destination is not a directory: $destination" >&2
    exit 2
fi

families=(
    "libggml.so"
    "libggml-base.so"
    "libggml-cpu.so"
    "libggml-hip.so"
)
patterns=(
    "libggml.so*"
    "libggml-base.so*"
    "libggml-cpu.so*"
    "libggml-hip.so*"
)
matches=()

for index in "${!families[@]}"; do
    family_matches=()
    mapfile -d '' -t family_matches < <(
        find "$source_root" \( -type f -o -type l \) \
            -name "${patterns[$index]}" -print0
    )

    if [[ ${#family_matches[@]} -eq 0 ]]; then
        echo "missing required library family: ${families[$index]}" >&2
        exit 1
    fi

    matches+=("${family_matches[@]}")
done

for library in "${matches[@]}"; do
    cp -P -- "$library" "$destination/"
done
