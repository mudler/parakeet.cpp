"""JSON (de)serialisation of the seg_eval results, replacing the old pickle.

Per recording the result is a list of [name, policy, trim, vector] rows, so no
key has to be joined into a string and a name may contain any character."""
import numpy as np


def res_to_json(res):
    return {rid: [[k[0], k[1], float(k[2]), np.asarray(v).tolist()] for k, v in rdict.items()]
            for rid, rdict in res.items()}


def res_from_json(obj):
    return {rid: {(name, pol, float(trim)): np.array(v) for name, pol, trim, v in rows}
            for rid, rows in obj.items()}
