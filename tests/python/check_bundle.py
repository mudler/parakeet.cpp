#!/usr/bin/env python3
"""scripts/bundle_gguf.py: build, --list, --verify, --notice and the refusals.

Model independent: it writes small synthetic GGUF files itself.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

try:
    import numpy as np
    from gguf import GGMLQuantizationType, GGUFWriter
except ImportError:
    print("skip: numpy and gguf are needed")
    sys.exit(77)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "bundle_gguf.py")
failures = 0


def check(cond, msg):
    global failures
    if not cond:
        print(f"FAIL: {msg}", file=sys.stderr)
        failures += 1


def run(*args):
    p = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def write_asr(path, arch="parakeet", parakeet_arch="tdt", license=None, extra_key=True):
    w = GGUFWriter(path, arch)
    w.add_string("general.name", "stand-in/asr")
    if license:
        w.add_string("general.license", license)
    if parakeet_arch:
        w.add_string("parakeet.arch", parakeet_arch)
    w.add_uint32("parakeet.encoder.d_model", 16)
    w.add_uint32("parakeet.vocab_size", 5)
    w.add_float32("parakeet.preprocessor.preemph", 0.97)
    w.add_bool("parakeet.encoder.xscaling", True)
    w.add_array("parakeet.tdt.durations", [0, 1, 2, 3, 4])
    w.add_array("parakeet.tokenizer.pieces", ["<unk>", "a", "b", "c", "d"])
    rng = np.random.default_rng(1)
    w.add_tensor("encoder.w", rng.standard_normal((4, 16)).astype(np.float32))
    w.add_tensor("encoder.h", rng.standard_normal((8, 8)).astype(np.float16))
    # a quantised-looking tensor: raw bytes with a block type
    q = rng.integers(0, 255, size=(4, 34), dtype=np.uint8)
    w.add_tensor("encoder.q", q, raw_shape=q.shape, raw_dtype=GGMLQuantizationType.Q8_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_vad(path, license="MIT"):
    w = GGUFWriter(path, "silero_vad")
    w.add_string("general.name", "silero-vad")
    if license:
        w.add_string("general.license", license)
    w.add_array("silero_vad.sample_rates", [16000, 8000])
    w.add_uint32("silero_vad.lstm.hidden", 128)
    rng = np.random.default_rng(2)
    w.add_tensor("vad16k.decoder.rnn.bias_ih", rng.standard_normal(32).astype(np.float32))
    w.add_tensor("vad8k.decoder.rnn.bias_ih", rng.standard_normal(32).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


ASR_META = {"kind": "asr", "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "source": "nvidia/stand-in", "attribution": "Stand-in model by NVIDIA", "changes": "Converted to GGUF"}
VAD_META = {"kind": "vad", "license": "MIT", "license_url": "https://github.com/snakers4/silero-vad/blob/master/LICENSE",
            "source": "https://github.com/snakers4/silero-vad", "attribution": "Copyright (c) 2020-present Silero Team",
            "changes": "Converted to GGUF"}


def manifest(path, comps, name="test-bundle", extra=None):
    m = {"name": name, "components": comps}
    if extra:
        m.update(extra)
    with open(path, "w") as f:
        json.dump(m, f)


def comp(name, file, meta, **over):
    c = {"name": name, "file": file, **meta}
    c.update(over)
    return c


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    d = tempfile.mkdtemp(prefix="check_bundle_")
    try:
        os.chdir(d)
        write_asr("asr.gguf")
        write_vad("vad.gguf")
        manifest("m.json", [comp("asr", "asr.gguf", ASR_META), comp("vad", "vad.gguf", VAD_META)])

        # --- roundtrip, determinism, verify against the sources ---
        rc, out, err = run("--out", "b1.gguf", "--manifest", "m.json")
        check(rc == 0, f"build failed: {err}")
        rc, out, err = run("--out", "b2.gguf", "--manifest", "m.json")
        check(rc == 0 and sha("b1.gguf") == sha("b2.gguf"), "the output is not deterministic")
        rc, out, err = run("--verify", "b1.gguf", "--source", "asr=asr.gguf", "--source", "vad=vad.gguf")
        check(rc == 0 and "OK" in out, f"verify with sources failed: {err}")
        rc, out, err = run("--verify", "b1.gguf")
        check(rc == 0, f"verify failed: {err}")

        rc, out, err = run("--list", "b1.gguf", "--json")
        j = json.loads(out) if rc == 0 else {}
        check(list(j.get("components", {})) == ["asr", "vad"], "list order or content")
        a = j.get("components", {}).get("asr", {})
        check(a.get("license") == "CC-BY-4.0" and a.get("tensors") == 3 and "Q8_0" in a.get("types", []), "list: asr info")
        check(a.get("source_sha256") == sha("asr.gguf"), "source_sha256 is not the input's sha256")
        rc, out, err = run("--list", "b1.gguf")
        check(rc == 0 and "asr:" in out and "vad:" in out, "list text")

        rc, out, err = run("--notice", "b1.gguf")
        check(rc == 0 and "CC-BY-4.0" in out and "Copyright (c) 2020-present Silero Team" in out and "NVIDIA" in out,
              "notice text lacks a licence or credit")
        check(sha("asr.gguf") in out, "notice lacks the input sha256")

        # --- a tampered bundle fails verify ---
        data = bytearray(open("b1.gguf", "rb").read())
        data[-5] ^= 0xFF
        open("tampered.gguf", "wb").write(bytes(data))
        rc, out, err = run("--verify", "tampered.gguf")
        check(rc == 1 and "content_sha256" in err, "a flipped tensor byte must fail verify")
        open("short.gguf", "wb").write(bytes(data[:-200]))
        rc, out, err = run("--verify", "short.gguf")
        check(rc != 0, "a truncated bundle must fail verify")
        open("junk.gguf", "wb").write(b"not a gguf")
        check(run("--verify", "junk.gguf")[0] == 1 and run("--list", "junk.gguf")[0] == 1, "junk must fail cleanly")
        check(run("--verify", "asr.gguf")[0] == 1, "a plain GGUF is not a bundle")
        # verify with the wrong source
        rc, out, err = run("--verify", "b1.gguf", "--source", "asr=vad.gguf")
        check(rc == 1, "a source that differs must fail verify")

        # --- refusals at build time ---
        def refuse(label, comps, needle, **kw):
            manifest("bad.json", comps, **kw)
            rc, out, err = run("--out", "bad.gguf", "--manifest", "bad.json")
            check(rc == 1 and needle in err, f"{label}: want a refusal with '{needle}', got rc={rc} {err!r}")
            check(not os.path.exists("bad.gguf") and not os.path.exists("bad.gguf.tmp"), f"{label}: left a file behind")

        for k in ("license", "license_url", "source", "attribution", "changes"):
            meta = {x: y for x, y in ASR_META.items() if x != k}
            refuse(f"missing {k}", [comp("asr", "asr.gguf", meta)], f"'{k}'")
            refuse(f"empty {k}", [comp("asr", "asr.gguf", ASR_META, **{k: "  "})], f"'{k}'")
        refuse("non-commercial", [comp("asr", "asr.gguf", ASR_META, license="CC-BY-NC-SA-4.0")], "must not be bundled")
        refuse("no derivatives", [comp("asr", "asr.gguf", ASR_META, license="CC-BY-ND-4.0")], "must not be bundled")
        refuse("bad licence id", [comp("asr", "asr.gguf", ASR_META, license="some licence")], "SPDX")
        refuse("bad licence url", [comp("asr", "asr.gguf", ASR_META, license_url="file:///x")], "http")
        write_asr("asr_lic.gguf", license="Apache-2.0")
        refuse("input licence conflicts with the manifest", [comp("asr", "asr_lic.gguf", ASR_META)], "declares general.license")
        write_asr("asr_lic_same.gguf", license="cc by 4.0")   # same licence, other spelling: allowed
        manifest("ok2.json", [comp("asr", "asr_lic_same.gguf", {**ASR_META, "license": "CC-BY-4.0"})])
        check(run("--out", "ok2.gguf", "--manifest", "ok2.json")[0] == 0, "an input with the same licence must be accepted")
        write_vad("vad_bad_lic.gguf", license="GPL-3.0")
        refuse("silero licence conflict", [comp("vad", "vad_bad_lic.gguf", VAD_META)], "declares general.license")
        refuse("kind does not match the file", [comp("asr", "vad.gguf", ASR_META)], "kind asr")
        refuse("vad kind on an ASR file", [comp("vad", "asr.gguf", VAD_META)], "kind vad")
        write_asr("diar.gguf", parakeet_arch="diarization")
        refuse("diarization is a later phase", [comp("asr", "diar.gguf", ASR_META)], "diarization")
        refuse("unknown kind", [comp("x", "asr.gguf", ASR_META, kind="diarization")], "kind must be")
        refuse("duplicate names", [comp("asr", "asr.gguf", ASR_META), comp("asr", "asr.gguf", ASR_META)], "duplicate")
        for bad in ("Asr", "a.b", "general", "parakeet", "bundle", "1a", ""):
            refuse(f"bad name {bad!r}", [comp(bad, "asr.gguf", ASR_META)], "component name")
        refuse("missing file", [comp("asr", "nope.gguf", ASR_META)], "does not exist")
        refuse("unknown component key", [comp("asr", "asr.gguf", ASR_META, colour="red")], "unknown keys")
        refuse("unknown manifest key", [comp("asr", "asr.gguf", ASR_META)], "unknown keys", extra={"x": 1})
        refuse("no components", [], "components")
        open("notgguf.gguf", "wb").write(b"hello")
        refuse("input is not a GGUF", [comp("asr", "notgguf.gguf", ASR_META)], "cannot read")
        refuse("input is a bundle", [comp("asr", "b1.gguf", ASR_META)], "already a bundle")
        write_asr("asr_noarch.gguf", parakeet_arch=None)
        refuse("input has no parakeet.arch", [comp("asr", "asr_noarch.gguf", ASR_META)], "kind asr")
        # a long component name makes a tensor name too long
        w = GGUFWriter("longname.gguf", "parakeet")
        w.add_string("parakeet.arch", "tdt"); w.add_uint32("parakeet.encoder.d_model", 4); w.add_uint32("parakeet.vocab_size", 5)
        w.add_tensor("t" * 60, np.zeros(4, dtype=np.float32))
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
        refuse("tensor name too long", [comp("asr", "longname.gguf", ASR_META)], "longer than")
        # the output must not be an input
        manifest("same.json", [comp("asr", "asr.gguf", ASR_META)])
        rc, out, err = run("--out", "asr.gguf", "--manifest", "same.json")
        check(rc == 1 and "also an input" in err, "output equal to an input must be refused")
        check(run("--verify", "b1.gguf", "--source", "asr=asr.gguf")[0] == 0, "the input was damaged by the refused build")

        # mixed licences in one bundle are fine and are reported per component
        check(a.get("license") != j["components"]["vad"]["license"], "the bundle mixes two licences")
        # a mode is required
        check(run()[0] != 0, "no mode must be an error")
    finally:
        os.chdir("/")
        shutil.rmtree(d, ignore_errors=True)
    if failures:
        print(f"{failures} check(s) failed", file=sys.stderr)
        return 1
    print("check_bundle OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
