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


def write_slice(path, license="CC-BY-4.0"):
    """A VAD-only slice of an ASR model (scripts/slice_vad_gguf.py): parakeet.arch is "vad"."""
    w = GGUFWriter(path, "parakeet")
    w.add_string("general.name", "stand-in/vad-head")
    if license:
        w.add_string("general.license", license)
    w.add_string("parakeet.arch", "vad")
    w.add_bool("parakeet.vad.present", True)
    w.add_uint32("parakeet.vad.d_in", 16)
    rng = np.random.default_rng(3)
    w.add_tensor("vad.w", rng.standard_normal((4, 16)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_diar(path, name="nvidia/Nemotron-3-Diarization"):
    w = GGUFWriter(path, "parakeet")
    w.add_string("general.name", name)
    w.add_string("parakeet.arch", "diarization")
    w.add_uint32("parakeet.encoder.d_model", 16)
    w.add_uint32("parakeet.diar.n_speakers", 4)
    rng = np.random.default_rng(4)
    w.add_tensor("encoder.w", rng.standard_normal((4, 16)).astype(np.float32))
    q = rng.integers(0, 255, size=(4, 34), dtype=np.uint8)
    w.add_tensor("encoder.q", q, raw_shape=q.shape, raw_dtype=GGMLQuantizationType.Q8_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_speechlm(path):
    w = GGUFWriter(path, "parakeet")
    w.add_string("general.name", "stand-in/speechlm-audio")
    w.add_string("parakeet.arch", "speechlm_audio")
    w.add_uint32("parakeet.encoder.d_model", 16)
    w.add_bool("parakeet.speechlm.present", True)
    w.add_uint32("parakeet.speechlm.llm_dim", 24)
    rng = np.random.default_rng(7)
    w.add_tensor("speechlm.proj.weight", rng.standard_normal((24, 16)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_ced(path):
    w = GGUFWriter(path, "ced")
    w.add_string("general.name", "mispeech/ced-tiny")
    w.add_string("ced.arch", "ced")
    w.add_uint32("ced.embed_dim", 8)
    w.add_array("ced.labels", ["Speech", "Music"])
    rng = np.random.default_rng(5)
    w.add_tensor("patch_embed.proj.weight", rng.standard_normal((8, 4)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_voice(path, arch="voicedetect", embedding=True, name="Wespeaker/wespeaker-voxceleb-resnet34-LM"):
    w = GGUFWriter(path, arch)
    w.add_string("general.name", name)
    w.add_string("voicedetect.arch", "wespeaker_resnet34")
    if embedding:
        w.add_uint32("voicedetect.embedding_dim", 256)
    rng = np.random.default_rng(6)
    w.add_tensor("model.seg_1.weight", rng.standard_normal((4, 8)).astype(np.float32))
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


DIAR_META = {"kind": "diar", "license": "OpenMDW-1.1", "license_url": "https://openmdw.ai/license/1-1/",
             "source": "nvidia/Nemotron-3-Diarization", "attribution": "Nemotron 3 Diarization by NVIDIA",
             "changes": "Converted to GGUF and quantised to Q8_0"}
CED_META = {"kind": "ced", "license": "Apache-2.0", "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
            "source": "mispeech/ced-tiny", "attribution": "CED-small by Heinrich Dinkel et al., Xiaomi (mispeech)", "changes": "Converted to GGUF"}
VOICE_META = {"kind": "voice", "license": "CC-BY-4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
              "source": "Wespeaker/wespeaker-voxceleb-resnet34-LM",
              "attribution": "WeSpeaker ResNet34-LM by the WeSpeaker project", "changes": "Converted from ONNX to GGUF"}
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

        # --- a VAD-only slice is a valid "vad" component, and not an "asr" one ---
        write_slice("slice.gguf")
        SLICE_META = dict(ASR_META, kind="vad")
        manifest("ms.json", [comp("asr", "asr.gguf", ASR_META), comp("vadh", "slice.gguf", SLICE_META)])
        rc, out, err = run("--out", "bs.gguf", "--manifest", "ms.json")
        check(rc == 0, f"a slice as kind vad must build: {err}")
        rc, out, err = run("--verify", "bs.gguf", "--source", "asr=asr.gguf", "--source", "vadh=slice.gguf")
        check(rc == 0, f"verify of a bundle with a slice failed: {err}")
        manifest("ms2.json", [comp("vadh", "slice.gguf", ASR_META)])
        rc, out, err = run("--out", "bs2.gguf", "--manifest", "ms2.json")
        check(rc != 0 and "VAD-only slice" in err, "a slice as kind asr must be refused")

        # --- the other kinds: diar, ced, voice; a full bundle with every kind ---
        write_diar("diar.gguf"); write_ced("ced.gguf"); write_voice("voice.gguf")
        manifest("mf.json", [comp("asr", "asr.gguf", ASR_META), comp("diar", "diar.gguf", DIAR_META),
                             comp("ced", "ced.gguf", CED_META), comp("voice", "voice.gguf", VOICE_META),
                             comp("vad", "vad.gguf", VAD_META)], name="full-test")
        rc, out, err = run("--out", "bf.gguf", "--manifest", "mf.json")
        check(rc == 0, f"full bundle failed to build: {err}")
        rc, out, err = run("--out", "bf2.gguf", "--manifest", "mf.json")
        check(rc == 0 and sha("bf.gguf") == sha("bf2.gguf"), "the full bundle is not deterministic")
        rc, out, err = run("--verify", "bf.gguf", "--source", "asr=asr.gguf", "--source", "diar=diar.gguf",
                           "--source", "ced=ced.gguf", "--source", "voice=voice.gguf", "--source", "vad=vad.gguf")
        check(rc == 0, f"verify of the full bundle failed: {err}")
        rc, out, err = run("--list", "bf.gguf", "--json")
        j2 = json.loads(out) if rc == 0 else {}
        check(sorted(j2.get("components", {})) == ["asr", "ced", "diar", "vad", "voice"], "full list components")
        check({c: v["kind"] for c, v in j2.get("components", {}).items()} ==
              {"asr": "asr", "diar": "diar", "ced": "ced", "voice": "voice", "vad": "vad"}, "full list kinds")
        # speechlm_audio: builds as its own kind, never as asr
        write_speechlm("slm.gguf")
        SLM_META = dict(DIAR_META, kind="speechlm_audio", source="nvidia/stand-in-speechlm", attribution="Stand-in by NVIDIA")
        manifest("msl.json", [comp("audio", "slm.gguf", SLM_META), comp("diar", "diar.gguf", DIAR_META)])
        rc, out, err = run("--out", "bsl.gguf", "--manifest", "msl.json")
        check(rc == 0, f"speechlm_audio + diar bundle failed to build: {err}")
        rc, out, err = run("--verify", "bsl.gguf", "--source", "audio=slm.gguf", "--source", "diar=diar.gguf")
        check(rc == 0, f"verify of the speechlm_audio bundle failed: {err}")
        manifest("msl2.json", [comp("audio", "slm.gguf", ASR_META)])
        rc, out, err = run("--out", "bsl2.gguf", "--manifest", "msl2.json")
        check(rc != 0 and "speechlm_audio" in err, "a speech LLM audio model as kind asr must be refused")
        manifest("msl3.json", [comp("audio", "asr.gguf", SLM_META)])
        rc, out, err = run("--out", "bsl3.gguf", "--manifest", "msl3.json")
        check(rc != 0 and "speechlm_audio" in err, "an ASR model as kind speechlm_audio must be refused")
        # NOTICE: every component's credit and the full text of each distinct licence
        rc, out, err = run("--notice", "bf.gguf")
        check(rc == 0, f"notice failed: {err}")
        for needle in ("Nemotron 3 Diarization by NVIDIA", "CED-small by Heinrich Dinkel et al., Xiaomi (mispeech)",
                       "WeSpeaker ResNet34-LM by the WeSpeaker project",
                       "Copyright (c) 2020-present Silero Team", "NVIDIA", "OpenMDW License Agreement, version 1.1",
                       "Apache License", "Attribution 4.0 International", "Permission is hereby granted",
                       "you shall retain in your distribution"):
            check(needle in out, f"notice lacks: {needle}")
        check(out.count("Licence text: ") == 4, "one text per distinct licence (CC-BY-4.0 is used twice)")
        # a licence without a shipped text is an error, not a silent gap
        manifest("mx.json", [comp("asr", "asr.gguf", dict(ASR_META, license="LicenseRef-Other"))])
        run("--out", "bx.gguf", "--manifest", "mx.json")
        rc, out, err = run("--notice", "bx.gguf")
        check(rc == 1 and "no licence text" in err, "a licence without a text must make --notice fail")

        # --- kinds must match the file, and the forbidden models are refused ---
        def refuse2(label, comps, needle):
            manifest("bad2.json", comps)
            rc, out, err = run("--out", "bad2.gguf", "--manifest", "bad2.json")
            check(rc == 1 and needle in err, f"{label}: want a refusal with '{needle}', got rc={rc} {err!r}")
            check(not os.path.exists("bad2.gguf"), f"{label}: left a file behind")
        refuse2("diar kind on an ASR file", [comp("diar", "asr.gguf", DIAR_META)], "kind diar")
        refuse2("ced kind on a voice file", [comp("ced", "voice.gguf", CED_META)], "kind ced")
        refuse2("voice kind on a ced file", [comp("voice", "ced.gguf", VOICE_META)], "kind voice")
        write_voice("analyze.gguf", embedding=False, name="audeering/wav2vec2-large-robust-24-ft-age-gender")
        refuse2("analysis head as voice (no embedding)", [comp("voice", "analyze.gguf", VOICE_META)], "kind voice")
        write_voice("analyze2.gguf", name="audeering/wav2vec2-large-robust-24-ft-age-gender")
        refuse2("audeering by file name", [comp("voice", "analyze2.gguf", VOICE_META)], "must not be bundled")
        refuse2("audeering by source and licence",
                [comp("voice", "voice.gguf", dict(VOICE_META, source="audeering/wav2vec2-large-robust", license="CC-BY-NC-SA-4.0"))],
                "must not be bundled")
        refuse2("NVIDIA Open Model License (EOU)",
                [comp("asr", "asr.gguf", dict(ASR_META, source="nvidia/parakeet_realtime_eou_120m-v1",
                                              license="LicenseRef-NVIDIA-Open-Model-License"))], "must not be bundled")
        refuse2("EOU by licence text", [comp("asr", "asr.gguf", dict(ASR_META, license="NVIDIA-Open-Model-License"))],
                "must not be bundled")
        refuse2("diar with a wrong licence", [comp("diar", "diar.gguf", dict(DIAR_META, license="CC-BY-4.0"))], "is licensed OpenMDW-1.1")
        refuse2("ced with a wrong licence", [comp("ced", "ced.gguf", dict(CED_META, license="MIT"))], "is licensed Apache-2.0")
        refuse2("wespeaker as Apache-2.0 (CC-BY-4.0 is required)",
                [comp("voice", "voice.gguf", dict(VOICE_META, license="Apache-2.0"))], "is licensed CC-BY-4.0")

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
        refuse("a diarization file is not kind asr", [comp("asr", "diar.gguf", ASR_META)], "use kind diar")
        refuse("unknown kind", [comp("x", "asr.gguf", ASR_META, kind="sound")], "kind must be")
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
