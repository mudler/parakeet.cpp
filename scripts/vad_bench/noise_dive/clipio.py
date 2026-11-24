"""JSON (de)serialisation of the e_set clips, replacing the old pickle.

numpy arrays are written as {"__ndarray__": [...], "dtype": "float64"} so the
dtype survives the round trip. float32 and float64 values are exact in JSON
(Python writes the shortest repr that parses back to the same double), so the
reload is lossless. numpy scalars are written as plain Python numbers."""
import json
import numpy as np


class ClipEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "dtype": obj.dtype.str}
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def _hook(d):
    if "__ndarray__" in d and "dtype" in d:
        return np.array(d["__ndarray__"], dtype=np.dtype(d["dtype"]))
    return d


def dump_clips(clips, path):
    with open(path, "w") as f:
        json.dump(clips, f, cls=ClipEncoder)


def load_clips(path):
    with open(path, "r") as f:
        return json.load(f, object_hook=_hook)
