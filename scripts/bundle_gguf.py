#!/usr/bin/env python3
"""Build, inspect and verify bundle GGUF files (format: docs/bundle.md).

A bundle merges single-model GGUF files into one file. No weight is converted or
re-quantised: every tensor is copied byte for byte with its original type, so a
component keeps the quantisation it was published with.

Build (the manifest is JSON; "file" is relative to the manifest):

  {"name": "parakeet-bundle-standard",
   "components": [
     {"name": "asr", "file": "asr.gguf", "kind": "asr",
      "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
      "source": "nvidia/parakeet-tdt-0.6b-v3",
      "attribution": "Parakeet TDT 0.6B v3 by NVIDIA, CC-BY-4.0",
      "changes": "Converted to GGUF and quantised to Q8_0"},
     {"name": "vad", "file": "silero-vad-f16.gguf", "kind": "vad", ...}]}

  bundle_gguf.py --out bundle.gguf --manifest manifest.json

Inspect and verify:

  bundle_gguf.py --list bundle.gguf [--json]
  bundle_gguf.py --verify bundle.gguf [--source asr=asr.gguf --source vad=silero.gguf]
  bundle_gguf.py --notice bundle.gguf

The output is deterministic: the same inputs and manifest give the same bytes.
The script refuses a component without full licence metadata, a licence that
contradicts the licence the input file declares, and non-commercial licences.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
from gguf import GGUFReader, GGUFValueType, GGUFWriter

ARCH = "parakeet-bundle"
VERSION = 1
KINDS = ("asr", "vad")          # kinds this tool can build (docs/bundle.md lists the reserved ones)
RESERVED_NAMES = {"general", "parakeet", "bundle"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
LICENSE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")
REQUIRED = ("license", "license_url", "source", "attribution", "changes")
MANIFEST_KEYS = {"name", "file", "kind", *REQUIRED}
MAX_TENSOR_NAME = 63            # ggml keeps names in 64 bytes including the NUL
SKIP_KEYS = {"general.alignment"}


class BundleError(Exception):
    pass


# --------------------------------------------------------------------------- helpers

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_bytes(t):
    return np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)


def content_digest(tensors):
    """sha256 over the tensors of one component, in file order.

    Per tensor: name (utf-8), 0x00, ggml type id (u32 LE), number of dims (u32 LE),
    each dim (u64 LE, GGUF order), then the raw data bytes."""
    h = hashlib.sha256()
    for name, ttype, dims, data in tensors:
        h.update(name.encode("utf-8") + b"\0")
        h.update(int(ttype).to_bytes(4, "little"))
        h.update(len(dims).to_bytes(4, "little"))
        for d in dims:
            h.update(int(d).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def reader_tensors(reader, prefix=""):
    """(name without prefix, type id, dims, data) for the tensors that start with prefix."""
    out = []
    for t in reader.tensors:
        if not t.name.startswith(prefix):
            continue
        out.append((t.name[len(prefix):], int(t.tensor_type), [int(x) for x in t.shape], tensor_bytes(t)))
    return out


def field_value(field):
    """(value, type, sub_type) ready for GGUFWriter.add_key_value."""
    t = field.types[0]
    if t == GGUFValueType.ARRAY:
        sub = field.types[-1]
        if sub == GGUFValueType.STRING:
            return [str(bytes(field.parts[i]), "utf-8") for i in field.data], t, sub
        return field.contents(), t, sub
    return field.contents(), t, None


def str_field(reader, key):
    f = reader.fields.get(key)
    if f is None or f.types[0] != GGUFValueType.STRING:
        return None
    return str(bytes(f.parts[-1]), "utf-8")


def open_gguf(path):
    try:
        return GGUFReader(path)
    except Exception as e:  # not a GGUF file, truncated, ...
        raise BundleError(f"{path}: cannot read as a GGUF file ({e})")


def norm_license(s):
    return re.sub(r"[\s_]+", "-", s.strip().lower())


def check_license_metadata(where, meta):
    for k in REQUIRED:
        v = meta.get(k)
        if not isinstance(v, str) or not v.strip():
            raise BundleError(f"{where}: missing or empty '{k}' (every component must carry its licence, source, "
                              f"attribution and changes)")
    if not LICENSE_RE.match(meta["license"]):
        raise BundleError(f"{where}: license '{meta['license']}' is not a SPDX-style identifier "
                          f"(use for example CC-BY-4.0, MIT, Apache-2.0 or LicenseRef-<name>)")
    if re.search(r"(^|-)(nc|nd)(-|$)|noncommercial|non-commercial", meta["license"].lower()):
        raise BundleError(f"{where}: license '{meta['license']}' restricts commercial use or derivatives; "
                          f"such models must not be bundled")
    if not re.match(r"^https?://", meta["license_url"]):
        raise BundleError(f"{where}: license_url must be an http(s) URL")


# --------------------------------------------------------------------------- build

def load_manifest(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as e:
        raise BundleError(f"{path}: cannot read manifest ({e})")
    if not isinstance(m, dict) or not isinstance(m.get("name"), str) or not m["name"].strip():
        raise BundleError("manifest needs a non-empty 'name'")
    unknown = set(m) - {"name", "components"}
    if unknown:
        raise BundleError(f"manifest: unknown keys {sorted(unknown)}")
    comps = m.get("components")
    if not isinstance(comps, list) or not comps:
        raise BundleError("manifest needs a non-empty 'components' list")
    base = os.path.dirname(os.path.abspath(path))
    seen = set()
    for c in comps:
        if not isinstance(c, dict):
            raise BundleError("each component must be a JSON object")
        unknown = set(c) - MANIFEST_KEYS
        if unknown:
            raise BundleError(f"component {c.get('name')!r}: unknown keys {sorted(unknown)}")
        n = c.get("name")
        if not isinstance(n, str) or not NAME_RE.match(n) or n in RESERVED_NAMES:
            raise BundleError(f"bad component name {n!r} (lower case letters, digits and _, starting with a letter; "
                              f"not one of {sorted(RESERVED_NAMES)})")
        if n in seen:
            raise BundleError(f"duplicate component name '{n}'")
        seen.add(n)
        if c.get("kind") not in KINDS:
            raise BundleError(f"component {n}: kind must be one of {list(KINDS)}, got {c.get('kind')!r}")
        if not isinstance(c.get("file"), str) or not c["file"]:
            raise BundleError(f"component {n}: 'file' is required")
        c["path"] = c["file"] if os.path.isabs(c["file"]) else os.path.join(base, c["file"])
        check_license_metadata(f"component {n}", c)
    return m


def check_input(name, kind, reader, meta):
    """Reject an input that is not what its kind says, or whose licence disagrees with the manifest."""
    arch = str_field(reader, "general.architecture")
    if arch is None:
        raise BundleError(f"component {name}: input has no general.architecture")
    if arch == ARCH:
        raise BundleError(f"component {name}: input is already a bundle; nesting is not supported")
    if kind == "asr":
        if arch != "parakeet" or "parakeet.arch" not in reader.fields:
            raise BundleError(f"component {name}: kind asr needs a parakeet ASR GGUF "
                              f"(general.architecture is '{arch}')")
        if str_field(reader, "parakeet.arch") == "diarization":
            raise BundleError(f"component {name}: input is a diarization model, not ASR (a later phase)")
        for k in ("parakeet.encoder.d_model", "parakeet.vocab_size"):
            if k not in reader.fields:
                raise BundleError(f"component {name}: input lacks the key {k}")
    elif kind == "vad":
        if arch != "silero_vad" or "silero_vad.sample_rates" not in reader.fields:
            raise BundleError(f"component {name}: kind vad needs a Silero VAD GGUF "
                              f"(general.architecture is '{arch}')")
    declared = str_field(reader, "general.license")
    if declared is not None and norm_license(declared) != norm_license(meta["license"]):
        raise BundleError(f"component {name}: the input file declares general.license '{declared}' but the manifest "
                          f"says '{meta['license']}'; resolve the conflict before bundling")
    if not reader.tensors:
        raise BundleError(f"component {name}: input has no tensors")
    seen = set()
    for t in reader.tensors:
        if t.name in seen:
            raise BundleError(f"component {name}: duplicate tensor name {t.name}")
        seen.add(t.name)
        if len(f"{name}.{t.name}") > MAX_TENSOR_NAME:
            raise BundleError(f"component {name}: tensor name {name}.{t.name} is longer than {MAX_TENSOR_NAME} bytes")


def build(manifest_path, out_path):
    m = load_manifest(manifest_path)
    comps = m["components"]
    inputs = []
    for c in comps:
        if not os.path.isfile(c["path"]):
            raise BundleError(f"component {c['name']}: input file {c['file']} does not exist")
        if os.path.exists(out_path) and os.path.samefile(out_path, c["path"]):
            raise BundleError("the output file is also an input")
        r = open_gguf(c["path"])
        check_input(c["name"], c["kind"], r, c)
        inputs.append(r)

    tmp = out_path + ".tmp"
    w = GGUFWriter(tmp, ARCH, use_temp_file=False)
    w.add_string("general.name", m["name"])
    w.add_string("general.license", "other")
    w.add_string("general.license.name", "per-component")
    w.add_uint32("parakeet.bundle.version", VERSION)
    w.add_array("parakeet.bundle.components", [c["name"] for c in comps])
    total = 0
    for c, r in zip(comps, inputs):
        n = c["name"]
        digest = content_digest(reader_tensors(r))
        w.add_string(f"parakeet.bundle.{n}.kind", c["kind"])
        for k in REQUIRED:
            w.add_string(f"parakeet.bundle.{n}.{k}", c[k])
        w.add_string(f"parakeet.bundle.{n}.source_sha256", sha256_file(c["path"]))
        w.add_string(f"parakeet.bundle.{n}.content_sha256", digest)
        for key, f in r.fields.items():
            if key.startswith("GGUF.") or key in SKIP_KEYS:   # GGUF.* are reader internals
                continue
            val, vt, sub = field_value(f)
            w.add_key_value(f"{n}.{key}", val, vt, sub)
        for t in r.tensors:
            w.add_tensor(f"{n}.{t.name}", t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
            total += int(t.n_bytes)
        print(f"{n}: {len(r.tensors)} tensors from {c['file']}", file=sys.stderr)
    try:
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file(progress=False)
        w.close()
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"wrote {out_path} ({total / 1e6:.1f} MB of tensor data)", file=sys.stderr)


# --------------------------------------------------------------------------- read

def components_of(reader):
    f = reader.fields.get("parakeet.bundle.components")
    if f is None or f.types[0] != GGUFValueType.ARRAY or f.types[-1] != GGUFValueType.STRING:
        raise BundleError("no parakeet.bundle.components string array")
    return field_value(f)[0]


def load_bundle(path):
    r = open_gguf(path)
    if str_field(r, "general.architecture") != ARCH:
        raise BundleError(f"{path}: not a bundle (general.architecture is not '{ARCH}')")
    f = r.fields.get("parakeet.bundle.version")
    if f is None:
        raise BundleError("no parakeet.bundle.version")
    ver = int(f.contents())
    if ver < 1 or ver > VERSION:
        raise BundleError(f"bundle format version {ver} is not supported (this tool knows {VERSION})")
    return r


def component_info(r, n):
    info = {}
    for k in ("kind",) + REQUIRED + ("source_sha256", "content_sha256"):
        info[k] = str_field(r, f"parakeet.bundle.{n}.{k}")
    tens = [t for t in r.tensors if t.name.startswith(n + ".")]
    info["tensors"] = len(tens)
    info["bytes"] = sum(int(t.n_bytes) for t in tens)
    info["types"] = sorted({t.tensor_type.name for t in tens})
    return info


def list_bundle(path, as_json):
    r = load_bundle(path)
    out = {"name": str_field(r, "general.name"), "version": int(r.fields["parakeet.bundle.version"].contents()),
           "file_bytes": os.path.getsize(path), "components": {}}
    for n in components_of(r):
        out["components"][n] = component_info(r, n)
    if as_json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    print(f"{out['name']}  (format v{out['version']}, {out['file_bytes'] / 1e6:.1f} MB)")
    for n, i in out["components"].items():
        print(f"  {n}: kind={i['kind']} licence={i['license']} tensors={i['tensors']} "
              f"{i['bytes'] / 1e6:.1f} MB types={','.join(i['types'])}")
        print(f"      source={i['source']}  content_sha256={i['content_sha256']}")


def verify_bundle(path, sources):
    r = load_bundle(path)
    names = components_of(r)
    problems = []
    if len(set(names)) != len(names):
        problems.append("duplicate component names")
    for n in names:
        if not NAME_RE.match(n) or n in RESERVED_NAMES:
            problems.append(f"bad component name '{n}'")
    owned = tuple(n + "." for n in names)
    allowed_top = tuple(["general.", "parakeet.bundle."]) + owned
    for key in r.fields:
        if key.startswith("GGUF."):
            continue
        if not key.startswith(allowed_top):
            problems.append(f"key outside any component: {key}")
    size = os.path.getsize(path)
    for t in r.tensors:
        if not t.name.startswith(owned):
            problems.append(f"tensor outside any component: {t.name}")
        if int(t.data_offset) + int(t.n_bytes) > size:
            problems.append(f"tensor {t.name} extends past the end of the file")
    for n in names:
        i = component_info(r, n)
        if i["kind"] is None:
            problems.append(f"{n}: no kind")
        for k in REQUIRED:
            if not i[k]:
                problems.append(f"{n}: missing '{k}'")
        if i["license"]:
            try:
                check_license_metadata(n, {k: (i[k] or "") for k in REQUIRED})
            except BundleError as e:
                problems.append(str(e))
        if not i["tensors"]:
            problems.append(f"{n}: no tensors")
        if f"{n}.general.architecture" not in r.fields:
            problems.append(f"{n}: no {n}.general.architecture key")
        got = content_digest(reader_tensors(r, n + "."))
        if i["content_sha256"] != got:
            problems.append(f"{n}: content_sha256 does not match the tensors (recorded {i['content_sha256']}, "
                            f"computed {got})")
    for spec in sources:
        n, _, src = spec.partition("=")
        if n not in names or not src:
            problems.append(f"--source {spec}: not a component=path pair of this bundle")
            continue
        problems += compare_with_source(r, n, src)
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        raise BundleError(f"{len(problems)} problem(s) in {path}")
    extra = f", {len(sources)} source(s) match byte for byte" if sources else ""
    print(f"OK: {path}: {len(names)} component(s), {len(r.tensors)} tensors{extra}")


def compare_with_source(r, n, src):
    problems = []
    if not os.path.isfile(src):
        return [f"{n}: source file {src} does not exist"]
    s = open_gguf(src)
    rec = str_field(r, f"parakeet.bundle.{n}.source_sha256")
    if rec != sha256_file(src):
        problems.append(f"{n}: source file sha256 differs from the recorded source_sha256")
    a = reader_tensors(s)
    b = reader_tensors(r, n + ".")
    if [x[:3] for x in a] != [x[:3] for x in b]:
        problems.append(f"{n}: tensor names, types or shapes differ from the source")
    else:
        for x, y in zip(a, b):
            if not np.array_equal(x[3], y[3]):
                problems.append(f"{n}: tensor {x[0]} bytes differ from the source")
    for key, f in s.fields.items():
        if key.startswith("GGUF.") or key in SKIP_KEYS:
            continue
        g = r.fields.get(f"{n}.{key}")
        if g is None:
            problems.append(f"{n}: key {key} missing in the bundle")
            continue
        v1, v2 = field_value(f)[0], field_value(g)[0]
        if not np.array_equal(np.asarray(v1, dtype=object), np.asarray(v2, dtype=object)):
            problems.append(f"{n}: key {key} differs from the source")
    return problems


def notice(path):
    r = load_bundle(path)
    lines = [
        "NOTICE",
        "",
        f"{str_field(r, 'general.name')} is a single file that holds several machine-learning models.",
        "Each model keeps its own licence and attribution, listed below. They are also stored in the",
        "file header under parakeet.bundle.<component>.*. The models were converted to GGUF and",
        "merged without further changes to their weights, except where a component says otherwise.",
        "The full text of each licence is at the URL given for it.",
        "",
    ]
    for n in components_of(r):
        i = component_info(r, n)
        lines += [
            f"Component {n} ({i['kind']})",
            f"  Model:       {i['source']}",
            f"  Licence:     {i['license']}  {i['license_url']}",
            f"  Credit:      {i['attribution']}",
            f"  Changes:     {i['changes']}",
            f"  Input file sha256: {i['source_sha256']}",
            "",
        ]
    sys.stdout.write("\n".join(lines))


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="bundle file to write (with --manifest)")
    ap.add_argument("--manifest", help="JSON manifest of the components")
    ap.add_argument("--list", metavar="BUNDLE", help="print the components of a bundle")
    ap.add_argument("--json", action="store_true", help="with --list: JSON output")
    ap.add_argument("--verify", metavar="BUNDLE", help="check a bundle; exit 1 on any problem")
    ap.add_argument("--source", action="append", default=[], metavar="COMP=FILE",
                    help="with --verify: also compare a component with its single-model file")
    ap.add_argument("--notice", metavar="BUNDLE", help="print a NOTICE text from the licence keys")
    args = ap.parse_args(argv)
    try:
        if args.list:
            list_bundle(args.list, args.json)
        elif args.verify:
            verify_bundle(args.verify, args.source)
        elif args.notice:
            notice(args.notice)
        elif args.out and args.manifest:
            build(args.manifest, args.out)
        else:
            ap.error("give --out and --manifest, or one of --list, --verify, --notice")
    except BundleError as e:
        print(f"bundle_gguf: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
