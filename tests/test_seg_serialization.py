import os
import json
import tempfile
import unittest

class TestSegSerialization(unittest.TestCase):
    def test_seg_json_serialization_deserialization(self):
        # Sample meta data
        meta = [
            {"id": "rec1", "domain": "vox", "split": "test", "dur": 10.0, "nonspeech": 2.0},
            {"id": "rec2", "domain": "ami", "split": "train", "dur": 20.0, "nonspeech": 5.0}
        ]

        # Sample original res dictionary structure as produced by seg_eval.py
        # Keys in res dict for each recording ID are tuples: (name, pol, trim)
        sample_vec1 = [10.0, 2.0, 8.0, 7.5, 3.0, 1.0, 2.0, 1.0, 0.0, 0.5, 0.2, 0.3, 0.0, 1.0]
        sample_vec2 = [20.0, 5.0, 15.0, 14.0, 5.0, 2.0, 3.0, 2.0, 1.0, 1.0, 0.5, 0.5, 0.1, 2.1]

        res_orig = {
            "rec1": {
                ("silero", "base", 0.3): sample_vec1,
                ("silero", "base", 0.0): sample_vec1,
            },
            "rec2": {
                ("ultra", "soft", 0.3): sample_vec2,
            }
        }

        # Simulate serialization in seg_eval.py
        res_json = {
            rid: {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rdict.items()}
            for rid, rdict in res_orig.items()
        }

        data_to_dump = dict(meta=meta, res=res_json)

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name
            json.dump(data_to_dump, tmp)

        try:
            # Simulate deserialization in seg_report.py
            with open(tmp_path, "r") as f:
                R = json.load(f)

            M = R["meta"]
            res_loaded = {
                rid: {
                    (k.split("|")[0], k.split("|")[1], float(k.split("|")[2])): v
                    for k, v in rdict.items()
                }
                for rid, rdict in R["res"].items()
            }

            # Verify metadata equality
            self.assertEqual(M, meta)

            # Verify dictionary structure and values match
            self.assertEqual(set(res_loaded.keys()), set(res_orig.keys()))
            for rid in res_orig:
                for k in res_orig[rid]:
                    self.assertIn(k, res_loaded[rid])
                    self.assertEqual(res_loaded[rid][k], res_orig[rid][k])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
