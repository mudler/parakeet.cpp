"""JSON form of the per-file error cache written by lib.file_errors (replaces pickle).

`units` is keyed by tuples, which JSON cannot hold, so it is stored as a list of
[key, value] pairs; `events` tuples become lists. Both are restored on load."""


def err_to_json(r):
    return dict(r, units=[[list(k), v] for k, v in r["units"].items()],
                events=[list(e) for e in r["events"]])


def err_from_json(r):
    return dict(r, units={tuple(k): v for k, v in r["units"]},
                events=[tuple(e) for e in r["events"]])
