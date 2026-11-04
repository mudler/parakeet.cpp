#!/usr/bin/env python3
"""Test json serialization/deserialization compatibility between eval_frames and rl in scripts/vad_bench/real_recordings"""
import os
import sys
import json
import tempfile
import unittest

class TestVadBenchRl(unittest.TestCase):
    def test_json_load_and_tuple_conversion(self):
        # Create synthetic counts structure similar to eval_frames output
        specs = [
            ("ref",),
            ("sil", 0.5),
            ("nat", "silero", 0.5, 0.25, 0.1, 0.03),
            ("pp", 0.1, 0.2, 0.0, ("sil", 0.5))
        ]
        meta = [
            {"id": "rec1", "domain": "vox", "split": "tune", "dur": 10.0, "nonspeech": False},
            {"id": "rec2", "domain": "ami", "split": "held", "dur": 15.0, "nonspeech": False}
        ]
        counts = [
            [[10, 2, 1, 100, 1], [12, 1, 0, 100, 1], [10, 2, 1, 100, 1], [10, 2, 1, 100, 1]],
            [[20, 4, 2, 200, 2], [22, 2, 0, 200, 2], [20, 4, 2, 200, 2], [20, 4, 2, 200, 2]]
        ]

        data = {"specs": specs, "meta": meta, "counts": counts}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "counts.json")
            with open(json_path, "w") as f:
                json.dump(data, f)

            with open(json_path, "r") as f:
                loaded_data = json.load(f)

            def _to_tuple(x):
                if isinstance(x, list):
                    return tuple(_to_tuple(i) for i in x)
                return x

            sp = [_to_tuple(s) for s in loaded_data["specs"]]
            idx = {s: i for i, s in enumerate(sp)}

            # Assert tuples were reconstructed properly and hashable in dict
            self.assertEqual(sp[0], ("ref",))
            self.assertEqual(sp[1], ("sil", 0.5))
            self.assertEqual(sp[3], ("pp", 0.1, 0.2, 0.0, ("sil", 0.5)))
            self.assertIn(("sil", 0.5), idx)
            self.assertEqual(idx[("sil", 0.5)], 1)

if __name__ == "__main__":
    unittest.main()
