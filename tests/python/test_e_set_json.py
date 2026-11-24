import json
import tempfile
import os
import numpy as np
import pytest

class ClipEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def test_e_set_json_serialization_deserialization():
    clips = [
        {
            'name': 'speech_00',
            'cond': 'clean',
            'kind': 'speech',
            'spans': [[0.5, 2.3]],
            'z': {
                'redux': np.array([1.2, 3.4, -0.5], dtype=np.float32),
                'ultra': np.array([0.1, 0.2, 0.3], dtype=np.float32)
            },
            'sil': np.array([0.9, 0.8, 0.1], dtype=np.float32),
            'e10': np.array([-20.5, -10.2, -30.0], dtype=np.float32)
        },
        {
            'name': 'noise_00',
            'cond': 'nz-white20',
            'kind': 'noise',
            'spans': [],
            'stretch': [0.0, 1.0],
            'z': {
                'redux': np.array([-1.0, -2.0], dtype=np.float32),
                'ultra': np.array([-0.5, -1.5], dtype=np.float32)
            },
            'sil': np.array([0.0, 0.1], dtype=np.float32),
            'e10': np.array([-40.0, -45.0], dtype=np.float32)
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = os.path.join(tmpdir, 'e_set.json')

        # Serialize
        with open(json_file, 'w') as f:
            json.dump(clips, f, cls=ClipEncoder)

        # Deserialize (using exact deserialization pattern as in fig_energy.py and e_eval.py)
        with open(json_file, 'r') as f:
            loaded_clips = json.load(f)

        for c in loaded_clips:
            c['z'] = {m: np.array(v, dtype=np.float32) for m, v in c['z'].items()}
            c['sil'] = np.array(c['sil'], dtype=np.float32)
            c['e10'] = np.array(c['e10'], dtype=np.float32)

        assert len(loaded_clips) == len(clips)

        for original, loaded in zip(clips, loaded_clips):
            assert original['name'] == loaded['name']
            assert original['cond'] == loaded['cond']
            assert original['kind'] == loaded['kind']
            assert original['spans'] == loaded['spans']

            if 'stretch' in original:
                assert original['stretch'] == loaded['stretch']

            assert set(original['z'].keys()) == set(loaded['z'].keys())
            for model in original['z']:
                np.testing.assert_allclose(original['z'][model], loaded['z'][model])

            np.testing.assert_allclose(original['sil'], loaded['sil'])
            np.testing.assert_allclose(original['e10'], loaded['e10'])
