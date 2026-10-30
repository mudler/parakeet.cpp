import os
import json
import tempfile
import numpy as np

def test_fusion_json_roundtrip():
    # Construct synthetic results dictionary in the new JSON format expected by run.py and analyze.py
    cfg_items = [
        {
            "rule": "silero",
            "p": [0.5],
            "h": "ultra",
            "ms": 0.1,
            "arr": [[[1, 2, 3, 4, 5]]]
        }
    ]
    lr_items = [
        {
            "h": "ultra",
            "kind": "lr2",
            "split": "f0",
            "out": [[[1, 2, 3, 4, 5]]],
            "coef": ([0.1, 0.2], 0.5)
        }
    ]
    meta_item = {
        "names": ["clip1"],
        "cond": ["clean"],
        "fold": [0],
        "snr_true": [99],
        "noise_only": [False],
        "snr_est": [25.0],
        "dur": [5.0]
    }

    res_json = {
        "cfg": cfg_items,
        "lr": lr_items,
        "meta": meta_item
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "results.json")
        with open(json_path, "w") as f:
            json.dump(res_json, f)

        # Deserialize back as analyze.py does
        R = json.load(open(json_path, "r"))
        CFG = {(item["rule"], tuple(item["p"]), item["h"], item["ms"]): np.array(item["arr"]) for item in R["cfg"]}
        LR = {(item["h"], item["kind"], item["split"]): (np.array(item["out"]), item["coef"]) for item in R["lr"]}
        M = R["meta"]

        # Assert correct keys and values
        key_cfg = ("silero", (0.5,), "ultra", 0.1)
        assert key_cfg in CFG
        assert isinstance(CFG[key_cfg], np.ndarray)
        assert CFG[key_cfg].shape == (1, 1, 5)

        key_lr = ("ultra", "lr2", "f0")
        assert key_lr in LR
        assert isinstance(LR[key_lr][0], np.ndarray)
        assert LR[key_lr][1] == [[0.1, 0.2], 0.5]

        assert M["names"] == ["clip1"]

def test_rows_json_roundtrip():
    # Construct synthetic rows dictionary as written by report.py
    S_data = {
        "Silero thr .5": np.zeros((10, 5))
    }
    ALL_data = {
        "ultra": S_data
    }

    rows_json = {h: {n: arr.tolist() for n, arr in S.items()} for h, S in ALL_data.items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        rows_path = os.path.join(tmpdir, "rows.json")
        with open(rows_path, "w") as f:
            json.dump(rows_json, f)

        loaded = json.load(open(rows_path, "r"))
        assert "ultra" in loaded
        assert "Silero thr .5" in loaded["ultra"]
        rows_arr = np.array(loaded["ultra"]["Silero thr .5"])
        assert rows_arr.shape == (10, 5)
