"""Round trip of the seg_eval results through the shipped JSON helpers."""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "vad_bench", "real_recordings"))
from segio import res_from_json, res_to_json  # noqa: E402


class TestSegSerialization(unittest.TestCase):
    def test_roundtrip_with_separator_in_names(self):
        v = np.arange(14, dtype=np.float64) / 3
        res = {
            "rec1": {("silero", "base", 0.3): v, ("a|b", "c|d|e", 0.0): v + 1},
            "rec|2": {("ultra", "soft", 0.3): v * 2},
        }
        loaded = res_from_json(json.loads(json.dumps(res_to_json(res))))
        self.assertEqual(set(loaded), set(res))
        for rid in res:
            self.assertEqual(set(loaded[rid]), set(res[rid]))
            for k in res[rid]:
                np.testing.assert_array_equal(loaded[rid][k], res[rid][k])


if __name__ == "__main__":
    unittest.main()
