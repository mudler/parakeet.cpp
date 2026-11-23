# Model licences

The GGUF files published at `mudler/parakeet-cpp-gguf` are converted from other
people's models, so each file family keeps the licence of its source model. The
parakeet.cpp code is MIT-licensed.

This table is the reference for `scripts/publish_hf.py`. The script has no default
licence: a model id that is missing from its `LICENSES` table is an error. The
check `tests/python/check_publish_hf_licences.py` (and
`python3 scripts/publish_hf.py --check-licences`) fails when this table and the
script differ, so change both together.

| Source checkpoint | Licence | Licence text | Notice to keep |
|---|---|---|---|
| `FermionResearch/Phonon-2.5` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> | Phonon-2.5, Copyright 2026 Fermion Research, derived from nvidia/parakeet-tdt-0.6b-v3 (c) NVIDIA Corporation. Training audio includes CHiME-6 (CC-BY-SA-4.0) and SPGISpeech (Kensho terms) |
| `FermionResearch/Phonon-2.5-English` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> | Phonon-2.5 English, Copyright 2026 Fermion Research, derived from nvidia/parakeet-tdt-0.6b-v3 (c) NVIDIA Corporation. Training audio includes CHiME-6 (CC-BY-SA-4.0) and SPGISpeech (Kensho terms) |
| `moondream/parakeet-redux` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `moondream/parakeet-ultra` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/Nemotron-3-Diarization` | OpenMDW 1.1 | <https://openmdw.ai/license/1-1/> |  |
| `nvidia/nemotron-3.5-asr-streaming-0.6b` | OpenMDW 1.1 | <https://openmdw.ai/license/1-1/> |  |
| `nvidia/nemotron-labs-unified-asr-0.6b` | OpenMDW 1.1 | <https://openmdw.ai/license/1-1/> | Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. Use of this model is governed by the OpenMDW License Agreement, version 1.1 |
| `nvidia/parakeet-ctc-0.6b` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/parakeet-ctc-1.1b` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/parakeet-rnnt-0.6b` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/parakeet-rnnt-1.1b` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/parakeet-tdt-0.6b-v2` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/parakeet-tdt-0.6b-v3` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/parakeet-tdt-1.1b` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/parakeet-tdt_ctc-1.1b` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/parakeet-tdt_ctc-110m` | CC-BY-4.0 | <https://creativecommons.org/licenses/by/4.0/> |  |
| `nvidia/parakeet_realtime_eou_120m-v1` | NVIDIA Open Model License | <https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/> | Licensed by NVIDIA Corporation under the NVIDIA Open Model License |
| `snakers4/silero-vad` | MIT | <https://github.com/snakers4/silero-vad/blob/master/LICENSE> | Copyright (c) 2020-present Silero Team. Released under the MIT license |

Notes:

- The Moondream Ultra and Redux files are converted here, not trained. Credit
  Moondream and NVIDIA (`parakeet-tdt-0.6b-v3`) when you use them. The Redux F16 and
  Q8_0 files are dequantized from the ternary weights.
- The Phonon-2.5 files (multilingual and English) are converted here, not trained. The
  weights are CC-BY-4.0 and the Fermion code is Apache-2.0. Credit Fermion Research and
  NVIDIA (`parakeet-tdt-0.6b-v3`) when you use them. The training audio of Phonon-2.5 includes
  CHiME-6 (CC-BY-SA-4.0) and SPGISpeech (Kensho terms); the upstream NOTICE is kept
  in `NOTICE-phonon-2.5.txt` in the Hugging Face repo. Whether the share-alike term of CHiME-6
  reaches the trained weights is a legal question that this project does not settle.
- The Silero VAD files are converted from the official ONNX model, not trained.
- Static card text for the files that `publish_hf.py` does not convert (diarization,
  Moondream, Silero VAD, and the Nemotron 3.5 usage note) lives in `scripts/hf_card/`.
