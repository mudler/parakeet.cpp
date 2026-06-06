#!/usr/bin/env python3
"""Dump NeMo Parakeet intermediate tensors to ``baseline.gguf`` for C++ parity.

Phase 1 validates each C++ stage (mel -> subsampling -> conformer layers ->
encoder -> CTC) by diffing against the exact tensors this script captures from
the reference NeMo model. Correctness and determinism are therefore paramount:

* ``dither`` is forced to 0.0 so the mel spectrogram is reproducible (the C++
  side skips dither too).
* The forward runs under ``torch.no_grad()`` with the model in ``eval()``.
* The CTC logits are produced by running the encoder + CTC head **explicitly**
  rather than via ``transcribe``. ``parakeet-tdt_ctc-110m`` is a hybrid
  (TDT + CTC) model whose default ``transcribe`` uses the RNNT/TDT head and
  never calls ``ctc_decoder`` — a forward hook on it would never fire. Driving
  ``m.ctc_decoder`` directly guarantees the CTC logits are real.

Stored tensors (squeezed; f32 except the int32 ids). Axis order documented in
``docs/conversion.md`` ("Baseline intermediates"):

* ``mel``             ``[n_mels, T]``      output of ``m.preprocessor``
* ``subsampling_out`` ``[T', d_model]``    output of ``m.encoder.pre_encode``
* ``pos_emb``         ``[2*T'-1, d_model]`` rel pos enc (``m.encoder.pos_enc`` out[1])
* ``enc_pre_layers``  ``[T', d_model]``    tensor fed INTO conformer ``layers[0]``
* ``l0_attn_in``      ``[T', d_model]``    self_attn input = ``norm_self_att(residual)``
                                           (NOTE: residual = enc_pre_layers + 0.5*FFN1(..),
                                           so this is NOT norm_self_att(enc_pre_layers))
* ``l0_attn_out``     ``[T', d_model]``    output of ``layers[0].self_attn`` (RelPosMHA)
* ``l0_conv_out``     ``[T', d_model]``    output of ``layers[0].conv`` (ConformerConvolution)
* ``enc_layer_0``     ``[T', d_model]``    output of conformer ``layers[0]``
* ``enc_layer_mid``   ``[T', d_model]``    output of conformer ``layers[n//2]``
* ``enc_layer_last``  ``[T', d_model]``    output of conformer ``layers[n-1]``
* ``encoder_out``     ``[d_model, T']``    output of ``m.encoder`` (transposed)
* ``ctc_logits``      ``[T', V+1]``        log-probs from ``m.ctc_decoder``
* ``ctc_argmax_ids``  ``[T']`` int32       argmax over the vocab axis of logits

Transducer-core ground truth (Phase 2; ``m.decoder`` = ``RNNTDecoder``
prediction net, ``m.joint`` = ``RNNTJoint`` TDT joint):

* ``pred_input_ids``  ``[U]`` int32        fixed label sequence fed to the
                                           prediction net (``[120, 7, 300, 42]``,
                                           all non-blank; blank=1024).
* ``pred_out``        ``[U+1, pred_hidden=640]``  output ``g`` of
                                           ``m.decoder.predict(y, add_sos=True)``,
                                           squeezed. **add_sos=True** so a zero
                                           "start-of-sequence" embedding step is
                                           PREPENDED: with U=4 input ids the
                                           output length is U+1 = 5. The SOS step
                                           uses the zero embedding row
                                           (``embed.weight[1024]`` is all-zero,
                                           ``padding_idx=blank=1024``). The C++
                                           prediction net MUST also prepend the
                                           SOS step to match.
* ``joint_out``       ``[N, U+1, 1030]``   RAW logits from
                                           ``m.joint.joint(enc_slice, pred_out)``
                                           with N = ``joint_enc_frames`` = 4
                                           leading encoder frames. **RAW logits,
                                           NOT log_softmax**: ``m.joint.joint`` on
                                           CPU defaults to applying
                                           ``log_softmax`` over all 1030 entries
                                           (``log_softmax`` cfg is ``None``), but
                                           the TDT greedy decoder consumes raw
                                           logits and splits them — token logits
                                           ``[:1025]`` (1024 vocab + blank) and
                                           duration logits ``[1025:]`` (the 5 TDT
                                           durations [0,1,2,3,4]) get SEPARATE
                                           log_softmaxes. So we force
                                           ``m.joint.log_softmax = False`` to dump
                                           the raw output that the C++ joint
                                           (plain ReLU + linear) produces.
* ``joint_enc_frames`` ``[1]`` int32       N = number of leading encoder frames
                                           used for ``joint_out`` (= 4). The C++
                                           joint test feeds ``encoder_out[:, :N]``.

TDT greedy ground truth (Phase 3; ``m`` with the transducer/TDT head selected):

* ``tdt_token_ids``    ``[L]`` int32        NeMo's TDT-head greedy decoded
                                            token-id sequence (``hyp.y_sequence``)
                                            for the clip. Captured by selecting
                                            the transducer head
                                            (``change_decoding_strategy(decoder_type='rnnt')``
                                            — for this hybrid the transducer head
                                            IS the TDT decoder) and running
                                            ``m.transcribe([audio])``. May be
                                            length-0 for a silent / tone clip.

String KVs:

* ``baseline.detok_text``  detokenized text for the ``detok_ids`` fixture
* ``baseline.ctc_text``    authoritative NeMo CTC greedy transcript of the clip
* ``baseline.tdt_text``    authoritative NeMo TDT-head greedy transcript of the clip

Pure-RNNT / streaming-EOU baseline (Phase 5; ``EncDecRNNTBPEModel`` with no CTC
head, e.g. ``nvidia/parakeet_realtime_eou_120m-v1``). When the model has no
``ctc_decoder`` the script dumps a smaller, RNNT-only baseline (the model's
normal offline forward already applies limited-context + causal + layer_norm):

* ``mel``, ``subsampling_out``, ``l0_conv_in``, ``l0_conv_out``,
  ``enc_layer_0``, ``encoder_out`` — the same encoder-stage tensors as above
  (from the hooks). ``l0_conv_in`` ``[T', d_model]`` is the INPUT fed into
  ``layers[0].conv`` (= ``norm_conv(residual)``), captured via a forward-pre-hook;
  it lets the C++ conv-module parity test exercise the conv sub-graph (layer_norm
  + causal depthwise conv) in ISOLATION from the chunked attention.
* ``rnnt_token_ids``  ``[L]`` int32   NeMo's RNNT greedy token-id sequence
                                      (``hyp.y_sequence``), INCLUDING the
                                      ``<EOU>``=1024 / ``<EOB>``=1025 special
                                      tokens the transducer emits.
* ``baseline.rnnt_token_count`` uint32  length of the above (so consumers can
                                        tell "empty" from "missing").
* ``baseline.rnnt_text``  str   the RNNT greedy transcript (raw, with ``<EOU>``).

Exit codes (ctest convention): 0 = ok, 2 = deps/model unavailable, 1 = fail.
"""
import argparse
import json
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

try:
    import gguf
except ImportError as e:  # pragma: no cover - env guard
    print(f"baseline: missing dependency 'gguf': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

try:
    import torch
    import soundfile as sf
    from nemo.collections.asr.models import ASRModel
except ImportError as e:  # pragma: no cover - env guard
    print(f"baseline: missing dependency: {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


def _squeeze(arr):
    """np.squeeze but never collapse to a 0-d scalar."""
    out = np.squeeze(np.asarray(arr))
    if out.ndim == 0:
        out = out.reshape(1)
    return np.ascontiguousarray(out)


def _get_md_flag(m, key):
    """Read a model_defaults entry (OmegaConf dict or attr access)."""
    md = getattr(m.cfg, "model_defaults", {}) or {}
    try:
        return md[key]
    except Exception:
        return getattr(md, key, None)


def _prompt_streaming_text(m, feats, feat_len, pidx, num_prompts, specials):
    """NeMo cache-aware *streaming* transcript for a prompt model, with the
    language prompt applied (the authoritative streaming target for the C++
    StreamingSession).

    Mirrors ``gen_stream_baseline.py`` for the encoder side, then applies the
    SAME prompt branch as the offline forward (transpose -> cat(onehot) ->
    prompt_kernel -> transpose) to the concatenated streamed encoder output, and
    decodes the whole prompt-conditioned streamed output in one shot (the
    cache-aware-equivalence property makes single-shot decode of the streamed
    output == per-chunk decode carrying state, which is what the C++ session does).

    Returns (stream_text, stream_token_ids[list]). Returns (None, None) when the
    encoder is not a cache-aware streaming encoder (so the caller can skip the KV).
    """
    import torch

    enc = m.encoder
    if not hasattr(enc, "cache_aware_stream_step"):
        return None, None
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    enc.setup_streaming_params()
    sc = enc.streaming_cfg

    sb = CacheAwareStreamingAudioBuffer(
        model=m, online_normalization=False, pad_and_drop_preencoded=False
    )
    sb.append_processed_signal(feats, stream_id=-1)
    sb_iter = iter(sb)
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        enc.get_initial_cache_state(batch_size=1)
    )

    outs = []
    for step_num, (chunk_audio, chunk_lengths) in enumerate(sb_iter):
        drop = sc.drop_extra_pre_encoded if step_num != 0 else 0
        keep_all = sb.is_buffer_empty()
        with torch.no_grad():
            (e, el, cache_last_channel, cache_last_time, cache_last_channel_len) = (
                enc.cache_aware_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=keep_all,
                    drop_extra_pre_encoded=drop,
                )
            )
        valid = int(el[0].item())
        outs.append(e[:, :, :valid].detach())  # [1, d_model, valid]

    stream_enc = torch.cat(outs, dim=2)  # [1, d_model, T']
    Tp = stream_enc.shape[2]
    stream_len = torch.tensor([Tp], dtype=torch.int64)

    # Apply the language prompt to the streamed encoder output (same branch as
    # the offline forward; one-hot constant over time).
    with torch.no_grad():
        encoded = stream_enc.transpose(1, 2)                      # [1, T', D]
        onehot = torch.zeros(1, Tp, num_prompts, dtype=encoded.dtype)
        onehot[:, :, pidx] = 1.0
        concat = torch.cat([encoded, onehot], dim=-1)             # [1, T', D+P]
        pk_out = m.prompt_kernel(concat)                          # [1, T', D]
        pk_enc = pk_out.transpose(1, 2).contiguous()             # [1, D, T']
        hyps = m.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=pk_enc, encoded_lengths=stream_len, return_hypotheses=True
        )
    first = hyps[0] if isinstance(hyps, list) else hyps
    if isinstance(first, list):
        first = first[0]
    ys = first.y_sequence
    ys = ys.cpu().tolist() if hasattr(ys, "cpu") else list(ys)
    stream_ids = [int(t) for t in ys]
    non_special = [t for t in stream_ids if t not in specials]
    stream_text = m.tokenizer.ids_to_text(non_special)
    return stream_text, stream_ids


def resolve_prompt_lang(m, lang=None):
    """Resolve (target_lang, prompt_index, num_prompts) for a prompt model.

    ``lang`` None/empty -> the model default ("auto" if present, else the first
    dictionary key). Exits(1) with PARAKEET_BASELINE_BAD_LANG on an unknown key.
    """
    md = m.cfg.model_defaults
    pdict = md.prompt_dictionary
    num_prompts = int(md.get("num_prompts", 128))
    default_lang = "auto" if "auto" in pdict else list(pdict.keys())[0]
    target_lang = lang or default_lang
    if target_lang not in pdict:
        keys = list(pdict.keys())
        print(
            f"PARAKEET_BASELINE_BAD_LANG: '{target_lang}' not in prompt_dictionary; "
            f"available (first 10): {keys[:10]}",
            file=sys.stderr,
        )
        sys.exit(1)
    return target_lang, int(pdict[target_lang]), num_prompts


def compute_prompt_reference(m, audio_path, lang=None):
    """NeMo reference for a prompt-conditioned (nemotron) model at target ``lang``.

    Runs the SAME prompt branch as NeMo's forward() (encoder -> transpose ->
    cat(onehot) -> prompt_kernel -> transpose) and decodes the prompt-conditioned
    encoder output via ``m.decoding.rnnt_decoder_predictions_tensor`` for BOTH the
    offline encoder pass and the cache-aware streaming pass. This is the
    authoritative path used by every Phase 2/3 baseline (the lhotse
    transcribe(target_lang=...) dataloader needs per-cut language metadata our bare
    wav fixtures lack).

    Returns a dict with: target_lang, prompt_index, num_prompts, encoder_out
    (np [D,T]), prompt_kernel_out (np [T,D]), rnnt_ids (np int32), rnnt_text,
    stream_text (str|None), stream_ids (list[int]|None).
    """
    import torch
    import soundfile as sf

    target_lang, pidx, num_prompts = resolve_prompt_lang(m, lang)

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        print(f"PARAKEET_BASELINE_BAD_AUDIO: expected 16k mono, got sr={sr}",
              file=sys.stderr)
        sys.exit(1)
    wt = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)  # [1, S]
    lt = torch.tensor([wt.shape[1]], dtype=torch.int64)

    with torch.no_grad():
        feats, flen = m.preprocessor(input_signal=wt, length=lt)
        enc, elen = m.encoder(audio_signal=feats, length=flen)  # [1, D, T]
        encoded = enc.transpose(1, 2)                            # [1, T, D]
        T = encoded.shape[1]
        onehot = torch.zeros(1, T, num_prompts, dtype=encoded.dtype)
        onehot[:, :, pidx] = 1.0
        concat = torch.cat([encoded, onehot], dim=-1)            # [1, T, D+P]
        pk_out = m.prompt_kernel(concat)                         # [1, T, D]

    # RNNT greedy reference for this language. We CANNOT use m.transcribe() here:
    # the prompt model's transcribe dataloader (LhotseSpeechToTextBpeDatasetWith
    # PromptIndex) resolves the prompt index from each cut's language metadata,
    # which our bare wav fixture lacks ("Unknown prompt key: 'None'"). Instead we
    # decode the prompt-conditioned encoder output DIRECTLY via the model's RNNT
    # decoding object — exactly what _transcribe_output_processing does, and the
    # same encoder-output the C++ engine feeds its rnnt_greedy after PromptKernel.
    pk_enc = pk_out.transpose(1, 2).contiguous()                 # [1, D, T] for decoding
    with torch.no_grad():
        hyps = m.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=pk_enc, encoded_lengths=elen, return_hypotheses=True
        )
    first = hyps[0] if isinstance(hyps, list) else hyps
    if isinstance(first, list):  # NBest -> take the top hypothesis
        first = first[0]
    ys = first.y_sequence
    ys = ys.cpu().tolist() if hasattr(ys, "cpu") else list(ys)
    rnnt_ids = np.array(list(ys), dtype=np.int32)
    rnnt_text = first.text if hasattr(first, "text") else str(first)

    # Resolve <EOU>/<EOB> special ids (if any) so the streaming transcript strips
    # them exactly as StreamingSession does (specials surface as events, not text).
    specials = set()
    for tok in ("<EOU>", "<EOB>"):
        try:
            ids = m.tokenizer.tokens_to_ids([tok])
            if ids and int(ids[0]) >= 0:
                specials.add(int(ids[0]))
        except Exception:
            pass

    # Cache-aware STREAMING transcript WITH the language prompt — the authoritative
    # target for the C++ StreamingSession. None if the model isn't a streaming
    # (cache-aware) encoder.
    stream_text, stream_ids = _prompt_streaming_text(
        m, feats, flen, pidx, num_prompts, specials
    )

    return {
        "target_lang": target_lang,
        "prompt_index": pidx,
        "num_prompts": num_prompts,
        "encoder_out": _squeeze(enc.cpu().float().numpy()),          # [D, T]
        "prompt_kernel_out": _squeeze(pk_out.cpu().float().numpy()),  # [T, D]
        "rnnt_ids": rnnt_ids,
        "rnnt_text": rnnt_text,
        "stream_text": stream_text,
        "stream_ids": stream_ids,
    }


def dump_prompt_baseline(m, args):
    """Dump the prompt-model baseline for a fixed target_lang:

      * ``encoder_out``       ``[D, T]``  RAW encoder output (BEFORE prompt).
      * ``prompt_kernel_out`` ``[T, D]``  prompt_kernel(cat([encoded, onehot])),
                                          i.e. NeMo's forward() prompt branch.
      * ``rnnt_token_ids``    ``[L]`` int32  NeMo RNNT greedy ids for this lang.
      * KVs: baseline.target_lang, baseline.prompt_index,
             baseline.rnnt_token_count, baseline.rnnt_text,
             baseline.stream_text (cache-aware streaming transcript WITH prompt,
             EOU/EOB stripped — the authoritative target for StreamingSession).

    Mirrors NeMo EncDecRNNTBPEModelWithPrompt.forward():
      encoded(B,D,T) -> transpose -> cat(onehot) -> prompt_kernel -> transpose.
    The one-hot is constant over time (one language per utterance).
    """
    ref = compute_prompt_reference(m, args.audio, args.lang)
    target_lang = ref["target_lang"]
    pidx = ref["prompt_index"]
    num_prompts = ref["num_prompts"]
    rnnt_ids = ref["rnnt_ids"]
    rnnt_text = ref["rnnt_text"]
    stream_text = ref["stream_text"]
    stream_ids = ref["stream_ids"]

    w = gguf.GGUFWriter(args.output, "parakeet-baseline-prompt")
    w.add_string("baseline.target_lang", target_lang)
    w.add_uint32("baseline.prompt_index", pidx)
    w.add_uint32("baseline.num_prompts", num_prompts)
    w.add_tensor("encoder_out", ref["encoder_out"])           # [D, T]
    w.add_tensor("prompt_kernel_out", ref["prompt_kernel_out"])  # [T, D]
    w.add_uint32("baseline.rnnt_token_count", int(rnnt_ids.shape[0]))
    if rnnt_ids.shape[0] > 0:
        w.add_tensor("rnnt_token_ids", np.ascontiguousarray(rnnt_ids))
    w.add_string("baseline.rnnt_text", rnnt_text)
    if stream_text is not None:
        w.add_string("baseline.stream_text", stream_text)
        sids = np.array(stream_ids, dtype=np.int32)
        w.add_uint32("baseline.stream_token_count", int(sids.shape[0]))
        if sids.shape[0] > 0:
            w.add_tensor("stream_token_ids", np.ascontiguousarray(sids))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: prompt baseline lang={target_lang} idx={pidx} "
        f"tokens={rnnt_ids.shape[0]} text={rnnt_text!r}"
    )
    if stream_text is not None:
        print(f"  baseline.stream_text={stream_text!r} (stream_tokens={len(stream_ids)})")
    else:
        print("  baseline.stream_text: SKIPPED (encoder is not cache-aware streaming)")


def _timestamps_decoding_cfg(m):
    """Build a decoding cfg (cloned from the model's own) that turns on
    per-frame/token/word confidence using the reproducible ``max_prob`` method.

    NeMo's ``max_prob`` is NOT the raw softmax probability: it is the *rescaled*
    confidence ``(p_max * V - 1) / (V - 1)`` (with ``alpha == 1.0``), where
    ``p_max = exp(max log-prob)`` and ``V == num_tokens == vocab_size + 1``
    (blank included; see ``asr_confidence_utils.get_confidence_measure_bank`` and
    ``ConfidenceMethodMixin._init_confidence_method`` where
    ``num_tokens = blank_id + 1``). We dump NeMo's actual confidence values so the
    C++ side can reproduce *that* mapping.

    IMPORTANT: ``method_cfg`` MUST pin ``alpha=1.0`` explicitly. The model's own
    ``decoding`` cfg carries a default ``confidence_method_cfg`` with
    ``alpha: 0.33`` (an entropy default). When we only set ``method_cfg.name``,
    OmegaConf MERGES our partial dict over that default, so the leftover
    ``alpha: 0.33`` survives — and NeMo's ``max_prob`` lambda has a SEPARATE
    ``t != 1.0`` branch ``((x.max·t).exp()·V^t - 1)/(V^t - 1)`` that is then used,
    yielding NOT the plain rescaled max-prob. Pinning ``alpha=1.0`` forces the
    intended ``(p_max·V - 1)/(V - 1)`` measure that this plan targets.

    ``aggregation='min'`` (NeMo default) -> per-token confidence over a CTC token's
    consecutive argmax run, and per-word confidence over its tokens, are the MIN.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(OmegaConf.to_container(m.cfg.decoding, resolve=True))
    cfg.confidence_cfg = {
        "preserve_frame_confidence": True,
        "preserve_token_confidence": True,
        "preserve_word_confidence": True,
        "exclude_blank": True,
        "aggregation": "min",
        "method_cfg": {"name": "max_prob", "alpha": 1.0},
    }
    return cfg


def _dump_head_timestamps(m, audio, decoder_type, blank_id):
    """Run one decoding head with timestamps+confidence and extract the per-token
    ``{id, frame, conf}`` arrays plus the per-word ``{w, start, end, conf}`` list.

    Returns (token_ids[int32], token_frames[int32], token_conf[f32], words[list],
             text, token_duration_or_None).

    Token-id sourcing differs by head:
      * RNNT/TDT: ``hyp.y_sequence`` IS the emitted token-id sequence.
      * CTC: ``hyp.y_sequence`` holds per-frame LOG-PROBS (shape ``[T, V+1]``) when
        timestamps/alignments are on, NOT ids. We reconstruct the ids exactly the
        way NeMo's CTC greedy does: argmax over the vocab axis, collapse repeats,
        drop blanks. This yields a sequence that aligns 1:1 with ``timestamp['char']``
        and ``token_confidence`` and detokenizes to ``hyp.text``.

    The per-token encoder frame differs by head (so the C++ greedy decoders can
    reproduce the frame they naturally know at emission):
      * CTC: ``timestamp['char'][i]['start_offset']`` — the frame where the
        collapsed token's consecutive argmax RUN STARTS (NeMo CTC start_offset =
        the previous token's emit frame; the C++ ctc_greedy records the run-start).
      * RNNT/TDT: ``timestamp['timestep'][i]`` — the RAW encoder frame ``time_idx``
        at which the symbol was emitted (== the C++ rnnt/tdt greedy ``t`` at
        emission). We deliberately do NOT use ``char.start_offset`` here: for TDT
        the char offsets pass through ``_refine_timestamps_tdt``, which overrides a
        PUNCTUATION token's start_offset to the previous token's end_offset — a
        word-timing refinement (Task 3), not the per-token emission frame.
    """
    import torch

    m.change_decoding_strategy(_timestamps_decoding_cfg(m), decoder_type=decoder_type)
    with torch.no_grad():
        out = m.transcribe([audio], batch_size=1, timestamps=True, return_hypotheses=True)
    hyp = (out[0] if isinstance(out, tuple) else out)[0]

    char_offsets = hyp.timestamp["char"]
    if decoder_type == "ctc":
        token_frames = np.array(
            [int(c["start_offset"]) for c in char_offsets], dtype=np.int32
        )
    else:
        # RNNT/TDT: raw emission frame (hypothesis.timestamp, stored under
        # 'timestep' after compute_rnnt_timestamps), filtered to non-blank tokens
        # to align 1:1 with the emitted token-id sequence.
        ts = hyp.timestamp
        timestep = ts["timestep"] if isinstance(ts, dict) else ts
        timestep = timestep.cpu().tolist() if hasattr(timestep, "cpu") else list(timestep)
        ys_for_frames = hyp.y_sequence
        ys_for_frames = (
            ys_for_frames.cpu().tolist()
            if hasattr(ys_for_frames, "cpu")
            else list(ys_for_frames)
        )
        token_frames = np.array(
            [int(s) for tok, s in zip(ys_for_frames, timestep) if tok != blank_id],
            dtype=np.int32,
        )
    token_conf = np.array([float(x) for x in hyp.token_confidence], dtype=np.float32)

    if decoder_type == "ctc":
        logits = np.asarray(hyp.y_sequence)  # [T, V+1] log-probs
        am = logits.argmax(-1)
        collapsed = []
        prev = -1
        for x in am.tolist():
            if x != prev:
                collapsed.append(x)
            prev = x
        token_ids = np.array(
            [x for x in collapsed if x != blank_id], dtype=np.int32
        )
        token_duration = None
    else:
        ys = hyp.y_sequence
        ys = ys.cpu().tolist() if hasattr(ys, "cpu") else list(ys)
        token_ids = np.array(list(ys), dtype=np.int32)
        td = getattr(hyp, "token_duration", None)
        if td is not None:
            token_duration = [int(x) for x in (td.tolist() if hasattr(td, "tolist") else td)]
        else:
            token_duration = None

    word_offsets = hyp.timestamp["word"]
    word_conf = [float(x) for x in hyp.word_confidence]
    if len(word_conf) != len(word_offsets):
        raise RuntimeError(
            f"baseline: {decoder_type} word_confidence ({len(word_conf)}) != "
            f"word_offsets ({len(word_offsets)})"
        )
    words = []
    for wo, wc in zip(word_offsets, word_conf):
        words.append(
            {
                "w": wo["word"],
                "start": round(float(wo["start"]), 6),
                "end": round(float(wo["end"]), 6),
                "conf": round(float(wc), 6),
            }
        )

    return token_ids, token_frames, token_conf, words, hyp.text, token_duration


def _run_timestamps(m, args):
    """``--timestamps`` mode: dump per-token + per-word timestamps and max_prob
    confidence for BOTH heads (TDT/RNNT + CTC) of the hybrid anchor model.

    Existing baseline dumper modes are untouched; this is a separate early path.
    """
    has_ctc = getattr(m, "ctc_decoder", None) is not None
    has_rnnt = getattr(m, "joint", None) is not None
    if not (has_ctc and has_rnnt):
        print(
            "PARAKEET_BASELINE_TS_NOT_HYBRID: --timestamps expects a hybrid "
            f"(CTC + TDT/RNNT) model; got has_ctc={has_ctc} has_rnnt={has_rnnt}",
            file=sys.stderr,
        )
        sys.exit(1)

    blank_id = int(
        getattr(getattr(m, "decoder", None), "blank_idx", None)
        or getattr(m.tokenizer, "vocab_size", 1024)
    )

    window_stride = float(m.cfg.preprocessor.window_stride)
    subsampling_factor = int(m.cfg.encoder.get("subsampling_factor", 8))
    frame_sec = window_stride * subsampling_factor  # 0.01 * 8 = 0.08 s/frame
    clip_sec = None

    # ---- TDT / transducer head ----
    (tdt_ids, tdt_frames, tdt_conf, tdt_words, tdt_text, tdt_dur) = _dump_head_timestamps(
        m, args.audio, "rnnt", blank_id
    )
    # ---- CTC head ----
    (ctc_ids, ctc_frames, ctc_conf, ctc_words, ctc_text, _ctc_dur) = _dump_head_timestamps(
        m, args.audio, "ctc", blank_id
    )

    # ---- Verification (fail loudly so a broken baseline is never written) ----
    def _verify(tag, ids, frames, conf, words, text):
        if not (len(ids) == len(frames) == len(conf)):
            raise RuntimeError(
                f"baseline[{tag}]: per-token arrays length mismatch: ids={len(ids)} "
                f"frames={len(frames)} conf={len(conf)}"
            )
        if len(frames) and not all(
            frames[i] <= frames[i + 1] for i in range(len(frames) - 1)
        ):
            raise RuntimeError(f"baseline[{tag}]: token frames not monotonic non-decreasing")
        if len(conf) and not all(0.0 < float(c) <= 1.0 + 1e-6 for c in conf):
            raise RuntimeError(f"baseline[{tag}]: token confidences not in (0, 1]")
        # word texts concatenate to the transcript
        joined = " ".join(w["w"] for w in words)
        if joined.split() != text.split():
            raise RuntimeError(
                f"baseline[{tag}]: word texts do not reconstruct transcript:\n"
                f"  words: {joined!r}\n  text:  {text!r}"
            )
        # word times sane: increasing starts, end>=start, within clip
        last = -1.0
        for w in words:
            if not (w["start"] >= last - 1e-6 and w["end"] >= w["start"] - 1e-6):
                raise RuntimeError(f"baseline[{tag}]: word times not sane: {w}")
            last = w["start"]
        # detok of ids == transcript
        detok = m.tokenizer.ids_to_text([int(i) for i in ids.tolist()])
        if detok != text:
            raise RuntimeError(
                f"baseline[{tag}]: detok(ids) != transcript:\n"
                f"  detok: {detok!r}\n  text:  {text!r}"
            )

    _verify("tdt", tdt_ids, tdt_frames, tdt_conf, tdt_words, tdt_text)
    _verify("ctc", ctc_ids, ctc_frames, ctc_conf, ctc_words, ctc_text)

    # frames must fit within the clip duration
    import soundfile as sf

    info = sf.info(args.audio)
    clip_sec = info.frames / info.samplerate
    for tag, frames in (("tdt", tdt_frames), ("ctc", ctc_frames)):
        if len(frames) and float(frames[-1]) * frame_sec > clip_sec + 1.0:
            raise RuntimeError(
                f"baseline[{tag}]: last token frame {frames[-1]} (>{frames[-1]*frame_sec:.2f}s) "
                f"exceeds clip duration {clip_sec:.2f}s"
            )

    # ---- Write the GGUF baseline ----
    w = gguf.GGUFWriter(args.output, "parakeet-baseline-ts")
    w.add_float32("baseline.frame_sec", float(frame_sec))
    w.add_uint32("baseline.subsampling_factor", int(subsampling_factor))

    w.add_tensor("ts_tdt_token_ids", np.ascontiguousarray(tdt_ids))
    w.add_tensor("ts_tdt_token_frames", np.ascontiguousarray(tdt_frames))
    w.add_tensor("ts_tdt_token_conf", np.ascontiguousarray(tdt_conf))
    # TDT per-token span = the predicted duration applied to the token
    # (hyp.token_duration). Used for word-end timing (Task 3) and to validate the
    # C++ tdt_greedy TokenInfo.span. Only present when NeMo reported durations.
    if tdt_dur is not None and len(tdt_dur) == len(tdt_ids) and len(tdt_dur) > 0:
        w.add_tensor(
            "ts_tdt_token_span", np.ascontiguousarray(np.array(tdt_dur, dtype=np.int32))
        )
    w.add_string("baseline.tdt_words_json", json.dumps(tdt_words, ensure_ascii=False))
    w.add_string("baseline.tdt_text", tdt_text)

    w.add_tensor("ts_ctc_token_ids", np.ascontiguousarray(ctc_ids))
    w.add_tensor("ts_ctc_token_frames", np.ascontiguousarray(ctc_frames))
    w.add_tensor("ts_ctc_token_conf", np.ascontiguousarray(ctc_conf))
    w.add_string("baseline.ctc_words_json", json.dumps(ctc_words, ensure_ascii=False))
    w.add_string("baseline.ctc_text", ctc_text)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # ---- Summary ----
    def _summary(tag, ids, frames, conf, words, dur):
        print(f"--- {tag} head ---")
        print(f"  tokens: {len(ids)}  (frame_sec={frame_sec:.3f}s, clip={clip_sec:.2f}s)")
        print(f"  first 6 token ids:    {ids[:6].tolist()}")
        print(f"  first 6 token frames: {frames[:6].tolist()}")
        print(f"  first 6 token conf:   {[round(float(c),4) for c in conf[:6]]}")
        if dur is not None:
            print(f"  first 6 token durs:   {dur[:6]}")
        print(f"  words: {len(words)}; first 5:")
        for wd in words[:5]:
            print(
                f"    {wd['w']!r:14} start={wd['start']:.2f}s end={wd['end']:.2f}s "
                f"conf={wd['conf']:.4f}"
            )

    print(f"baseline.frame_sec={frame_sec} subsampling_factor={subsampling_factor}")
    _summary("TDT/RNNT", tdt_ids, tdt_frames, tdt_conf, tdt_words, tdt_dur)
    _summary("CTC", ctc_ids, ctc_frames, ctc_conf, ctc_words, None)
    print(f"baseline.tdt_text: {tdt_text!r}")
    print(f"baseline.ctc_text: {ctc_text!r}")
    print(
        f"wrote {args.output}: timestamps+confidence baseline "
        f"(max_prob, aggregation=min, dither=0.0)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt_ctc-110m",
                    help="HF id or local .nemo")
    ap.add_argument("--audio", required=True, help="16k mono wav clip")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--timestamps",
        action="store_true",
        help="dump per-token/word timestamps + max_prob confidence for both "
        "heads (TDT/RNNT + CTC) instead of the encoder-stage baseline.",
    )
    ap.add_argument(
        "--att-context-size",
        type=int,
        default=None,
        help="if set to W, switch the encoder to NeMo local (Longformer) "
        "attention via change_attention_model('rel_pos_local_attn', [W, W]) "
        "before the forward, so the dumped pos_emb (2W+1), l0_attn_in/out and "
        "transcripts reflect banded local attention. Anchors the C++ "
        "banded-attention parity tests at NeMo quality.",
    )
    ap.add_argument(
        "--lang",
        default=None,
        help="target_lang for prompt models (default: model default / auto)",
    )
    args = ap.parse_args()

    is_local = pathlib.Path(args.model).exists()
    try:
        if is_local:
            m = ASRModel.restore_from(args.model, map_location="cpu")
        else:
            m = ASRModel.from_pretrained(args.model, map_location="cpu")
    except Exception as e:  # pragma: no cover - network/cache guard
        print(f"PARAKEET_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)

    m.eval()
    # Determinism: zero the spectrogram dither so the mel is reproducible.
    m.preprocessor.featurizer.dither = 0.0

    # Optional: switch to NeMo local (Longformer) attention so the dumped
    # baseline anchors the C++ banded-attention path. Must run BEFORE the hooks
    # below, since change_attention_model swaps the pos_enc and self_attn modules
    # (a hook registered on the old module would never fire). Mirrors the v3
    # model card's long-audio recipe.
    if args.att_context_size is not None:
        w = args.att_context_size
        m.change_attention_model("rel_pos_local_attn", [w, w])
        try:
            m.change_subsampling_conv_chunking_factor(1)
        except Exception as e:  # pragma: no cover - older NeMo without the API
            print(f"note: change_subsampling_conv_chunking_factor skipped: {e}",
                  file=sys.stderr)
        print(f"local attention: rel_pos_local_attn att_context_size=[{w},{w}]",
              file=sys.stderr)

    # --timestamps: dump per-token/word timestamps + max_prob confidence for both
    # heads, then return. Kept as a separate early path so the encoder-stage
    # baseline behaviour below is completely untouched.
    if args.timestamps:
        _run_timestamps(m, args)
        return

    # Prompt-conditioned multilingual model (nemotron, EncDecRNNTBPEModelWithPrompt):
    # dump the raw encoder output, the prompt_kernel projection, and the per-language
    # RNNT greedy reference. Kept as a separate early path; the encoder-stage hook
    # baseline below is for the hybrid / pure-RNNT models and is untouched.
    has_prompt = bool(_get_md_flag(m, "initialize_prompt_feature")) and (
        getattr(m, "prompt_kernel", None) is not None
    )
    if has_prompt:
        dump_prompt_baseline(m, args)
        return

    # Per-layer / module captures via forward hooks. The preprocessor and
    # encoder return (tensor, length) tuples; conformer layers return a bare
    # tensor (no cache) — handle both.
    cap = {}

    def save(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                cap[name] = t.detach().cpu().float().numpy()
        return fn

    def save_pos_emb(name):
        # RelPositionalEncoding.forward returns (dropout(x), pos_emb); we want the
        # SECOND element (the [1, 2*T'-1, d_model] relative positional encoding),
        # NOT element 0 (the scaled input embeddings).
        def fn(mod, inp, out):
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError(
                    f"baseline: expected pos_enc to return a (x, pos_emb) tuple, "
                    f"got {type(out).__name__} (len={len(out) if hasattr(out, '__len__') else 'n/a'})"
                )
            t = out[1]
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(
                    f"baseline: pos_enc out[1] is not a tensor (got {type(t).__name__})"
                )
            cap[name] = t.detach().cpu().float().numpy()
        return fn

    def save_layer_input(name):
        # Forward PRE-hook on layers[0]: capture the tensor fed INTO the layer
        # (encoder embeddings after subsampling + xscaling + dropout). The encoder
        # calls the layer with x as a KEYWORD arg, so prefer kwargs['x'] and fall
        # back to the first positional arg.
        def fn(mod, args, kwargs):
            t = kwargs.get("x") if "x" in kwargs else (args[0] if args else None)
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(
                    f"baseline: could not capture {name}: layer input was not a "
                    f"tensor (args={len(args)}, kwargs={sorted(kwargs)})"
                )
            cap[name] = t.detach().cpu().float().numpy()
        return fn

    def save_attn_in(name):
        # Forward PRE-hook on layers[0].self_attn: capture the `query` argument,
        # i.e. the normalized attention input (== `key` == `value`). The conformer
        # layer calls self_attn with query/key/value as KEYWORD args.
        def fn(mod, args, kwargs):
            t = kwargs.get("query") if "query" in kwargs else (args[0] if args else None)
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(
                    f"baseline: could not capture {name}: self_attn query was not a "
                    f"tensor (args={len(args)}, kwargs={sorted(kwargs)})"
                )
            cap[name] = t.detach().cpu().float().numpy()
        return fn

    def save_conv_in(name):
        # Forward PRE-hook on layers[0].conv (ConformerConvolution): capture the
        # INPUT to the conv module, i.e. norm_conv(residual). The conformer layer
        # calls self.conv(x, pad_mask=..., cache=...) so x is the first positional
        # arg (occasionally a kwarg). Squeeze to [T, d_model]. This lets the C++
        # conv-module parity test exercise the conv sub-graph in ISOLATION
        # (decoupled from the chunked attention, which is a later task).
        def fn(mod, args, kwargs):
            t = kwargs.get("x") if "x" in kwargs else (args[0] if args else None)
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(
                    f"baseline: could not capture {name}: conv input was not a "
                    f"tensor (args={len(args)}, kwargs={sorted(kwargs)})"
                )
            cap[name] = t.detach().cpu().float().numpy()
        return fn

    def submodule(path):
        """Resolve a dotted attribute path under ``m``; clear error if missing."""
        obj = m
        for attr in path.split("."):
            if attr.endswith("]") and "[" in attr:
                base, idx = attr[:-1].split("[")
                obj = getattr(obj, base)[int(idx)]
            else:
                obj = getattr(obj, attr, None)
            if obj is None:
                raise RuntimeError(
                    f"baseline: submodule '{path}' not found on the model "
                    f"(failed at '{attr}'). The NeMo module layout may differ."
                )
        return obj

    n = len(m.encoder.layers)
    handles = [
        m.preprocessor.register_forward_hook(save("mel")),
        m.encoder.pre_encode.register_forward_hook(save("subsampling_out")),
        # Fine-grained encoder captures for relpos-attention / conformer parity.
        submodule("encoder.pos_enc").register_forward_hook(save_pos_emb("pos_emb")),
        submodule("encoder.layers[0]").register_forward_pre_hook(
            save_layer_input("enc_pre_layers"), with_kwargs=True
        ),
        # self_attn input = norm_self_att(residual) where residual already includes
        # FFN1. Capturing it directly lets the relpos-attention parity test feed the
        # exact normalized input without re-implementing FFN1 (that's the next task).
        submodule("encoder.layers[0].self_attn").register_forward_pre_hook(
            save_attn_in("l0_attn_in"), with_kwargs=True
        ),
        submodule("encoder.layers[0].self_attn").register_forward_hook(
            save("l0_attn_out")
        ),
        # conv module INPUT (norm_conv(residual)) + OUTPUT for isolated conv parity.
        submodule("encoder.layers[0].conv").register_forward_pre_hook(
            save_conv_in("l0_conv_in"), with_kwargs=True
        ),
        submodule("encoder.layers[0].conv").register_forward_hook(save("l0_conv_out")),
        m.encoder.layers[0].register_forward_hook(save("enc_layer_0")),
        m.encoder.layers[n // 2].register_forward_hook(save("enc_layer_mid")),
        m.encoder.layers[n - 1].register_forward_hook(save("enc_layer_last")),
        m.encoder.register_forward_hook(save("encoder_out")),
    ]

    # Load the clip as float32 mono [1, num_samples].
    wav, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        print(f"PARAKEET_BASELINE_BAD_AUDIO: expected 16k mono, got sr={sr}",
              file=sys.stderr)
        sys.exit(1)
    wav_t = torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0)  # [1, S]
    len_t = torch.tensor([wav_t.shape[1]], dtype=torch.int64)

    # A pure RNNT model (e.g. the streaming nvidia/parakeet_realtime_eou_120m-v1,
    # an EncDecRNNTBPEModel) has no CTC head and no hybrid decoder-strategy switch.
    # For those we dump the shared encoder-stage tensors plus the RNNT greedy
    # token ids (incl. <EOU>/<EOB>); the CTC/TDT-specific captures below are
    # skipped. Existing hybrid (CTC + TDT) behaviour is left intact.
    has_ctc = getattr(m, "ctc_decoder", None) is not None
    is_pure_rnnt = (not has_ctc) and getattr(m, "joint", None) is not None

    # Explicit forward path (NOT transcribe) so the CTC head actually runs (for
    # hybrid models). The streaming/causal/layer_norm offline forward of the EOU
    # model is exactly this same encoder forward.
    #   preprocessor.forward(input_signal, length) -> (feats[B,n_mels,T], feat_len)
    #   encoder.forward(audio_signal, length)      -> (enc[B,d_model,T'], enc_len)
    #   ctc_decoder.forward(encoder_output)        -> log-probs [B, T', V+1]
    with torch.no_grad():
        feats, feat_len = m.preprocessor(input_signal=wav_t, length=len_t)
        enc, enc_len = m.encoder(audio_signal=feats, length=feat_len)
        ctc_log = m.ctc_decoder(encoder_output=enc) if has_ctc else None

    for h in handles:
        h.remove()

    # ---- Pure-RNNT (streaming EOU) baseline: encoder stages + RNNT greedy ----
    if is_pure_rnnt:
        # m.transcribe runs the model's normal offline forward (limited-context
        # + causal + layer_norm) and the RNNT greedy decoder. hyp.y_sequence is
        # the exact token-id sequence (incl. <EOU>=1024 / <EOB>=1025) the C++
        # rnnt_greedy loop must reproduce.
        with torch.no_grad():
            rnnt_out = m.transcribe([args.audio], batch_size=1, return_hypotheses=True)
        rnnt_hyps = rnnt_out[0] if isinstance(rnnt_out, tuple) else rnnt_out
        rnnt_first = rnnt_hyps[0]
        ys = rnnt_first.y_sequence
        if isinstance(ys, torch.Tensor):
            ys = ys.cpu().tolist()
        rnnt_token_ids = np.array(list(ys), dtype=np.int32)
        rnnt_text = rnnt_first.text if hasattr(rnnt_first, "text") else str(rnnt_first)

        # Required encoder-stage tensors for the offline-parity tests (Tasks 2-3):
        # mel, subsampling_out, l0_conv_out, enc_layer_0, encoder_out are already
        # captured by the hooks above. Write only those + the RNNT ground truth.
        keep = {"mel", "subsampling_out", "l0_conv_in", "l0_conv_out",
                "enc_layer_0", "encoder_out"}
        w = gguf.GGUFWriter(args.output, "parakeet-baseline")
        shapes = {}
        for k in sorted(cap):
            if k in keep:
                arr = _squeeze(cap[k])
                w.add_tensor(k, arr)
                shapes[k] = tuple(arr.shape)
        w.add_uint32("baseline.rnnt_token_count", int(rnnt_token_ids.shape[0]))
        if rnnt_token_ids.shape[0] > 0:
            w.add_tensor("rnnt_token_ids", np.ascontiguousarray(rnnt_token_ids))
            shapes["rnnt_token_ids"] = tuple(rnnt_token_ids.shape)
        w.add_string("baseline.rnnt_text", rnnt_text)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        print("baseline tensors:", shapes)
        print(f"baseline.rnnt_text: {repr(rnnt_text)}")
        print(f"rnnt_token_ids ({rnnt_token_ids.shape[0]}): {rnnt_token_ids.tolist()}")
        print(f"wrote {args.output}: tensors={len(shapes)} "
              f"(pure RNNT / streaming EOU, dither=0.0, explicit forward)")
        return

    cap["ctc_logits"] = ctc_log.detach().cpu().float().numpy()  # [B, T', V+1]

    # ctc_argmax_ids: greedy CTC ids = argmax over the vocab axis (last) of the
    # squeezed [T', V+1] logits. Deterministic end-to-end CTC check for Phase 1.
    logits = _squeeze(cap["ctc_logits"])  # [T', V+1]
    ids = (logits.argmax(-1) if logits.ndim == 2 else logits.argmax(0)).astype(np.int32)

    # ---- Transducer-core ground truth (Phase 2): prediction net + joint ----
    # Done BEFORE change_decoding_strategy() (which only swaps the decoding
    # strategy, not module weights, but we keep the model untouched to be safe).
    #
    # Prediction net: m.decoder.predict(y, state, add_sos, batch_size) returns
    # (g, hidden). With add_sos=True (the predict() default and what the RNNT/TDT
    # decoders use to prime the network), a zero "start-of-sequence" embedding
    # step is PREPENDED, so g is [B, U+1, pred_hidden]. blank=1024 is the
    # padding_idx so embed.weight[1024] is all-zero (== the SOS embedding).
    pred_input_ids = np.array([120, 7, 300, 42], dtype=np.int32)  # all non-blank
    add_sos = True
    y = torch.from_numpy(pred_input_ids.astype(np.int64)).unsqueeze(0)  # [1, U]
    with torch.no_grad():
        g, _hid = m.decoder.predict(y, state=None, add_sos=add_sos, batch_size=None)
    cap["pred_out"] = g.detach().cpu().float().numpy()  # [1, U+1, pred_hidden]

    # Sanity: SOS step (g row 0) uses the zero embedding -> confirm embed[blank]
    # is all-zero so the C++ side can reproduce the SOS step exactly.
    embed_w = m.decoder.prediction["embed"].weight.detach()
    blank_idx = int(getattr(m.decoder, "blank_idx", embed_w.shape[0] - 1))
    sos_embed_is_zero = bool(torch.all(embed_w[blank_idx] == 0).item())
    if not sos_embed_is_zero:  # pragma: no cover - layout guard
        print(
            f"PARAKEET_BASELINE_WARN: embed.weight[{blank_idx}] (SOS/padding) is "
            f"NOT all-zero (absmax={float(embed_w[blank_idx].abs().max())}); the "
            f"C++ SOS step must use this exact embedding row.",
            file=sys.stderr,
        )

    # Joint: m.joint.joint(f, g) with f=[B,T,enc_hidden], g=[B,U,pred_hidden] ->
    # [B, T, U, vocab+1+num_durations]=[B,T,U,1030]. `enc` from the encoder above
    # is [B, d_model, T'] so transpose to [B, T', d_model] first. Slice to the
    # first N=joint_enc_frames encoder frames to keep the dump small.
    #
    # RAW LOGITS: m.joint.joint() on CPU applies log_softmax over all 1030 entries
    # by default (joint.log_softmax cfg is None). But the TDT greedy decoder uses
    # RAW logits and splits them — token logits [:1025] and the 5 duration logits
    # [1025:] get SEPARATE log_softmaxes. The C++ joint emits raw logits, so we
    # force log_softmax=False here to dump the matching raw output.
    joint_enc_frames = 4
    enc_btd = enc.transpose(1, 2)  # [B, T', d_model]
    n_frames = min(joint_enc_frames, enc_btd.shape[1])
    enc_slice = enc_btd[:, :n_frames, :].contiguous()  # [1, N, enc_hidden]
    saved_log_softmax = m.joint.log_softmax
    try:
        m.joint.log_softmax = False  # raw logits, matching the C++ joint
        with torch.no_grad():
            jout = m.joint.joint(enc_slice, g)  # [1, N, U+1, 1030] raw logits
    finally:
        m.joint.log_softmax = saved_log_softmax
    cap["joint_out"] = jout.detach().cpu().float().numpy()  # [1, N, U+1, 1030]
    joint_enc_frames_arr = np.array([n_frames], dtype=np.int32)

    # ---- TDT greedy ground truth (Phase 3) ----
    # Select the transducer head (for this hybrid the transducer head IS the TDT
    # decoder) and run the model's own greedy decode to capture the reference
    # token-id sequence. change_decoding_strategy(decoder_type='rnnt') sets
    # cur_decoder='rnnt'; transcribe() then runs the TDT greedy path. We capture
    # hyp.y_sequence (the emitted token ids) — this is the EXACT sequence the C++
    # pk::tdt_greedy loop must reproduce. For a tone/silent clip the sequence may
    # be empty (length 0), which is fine.
    m.change_decoding_strategy(decoder_type="rnnt")
    assert m.cur_decoder == "rnnt", f"expected RNNT/TDT decoder, got {m.cur_decoder}"
    with torch.no_grad():
        tdt_out = m.transcribe([args.audio], batch_size=1, return_hypotheses=True)
    tdt_hyps = tdt_out[0] if isinstance(tdt_out, tuple) else tdt_out
    tdt_first = tdt_hyps[0]
    y_seq = tdt_first.y_sequence
    if isinstance(y_seq, torch.Tensor):
        y_seq = y_seq.cpu().tolist()
    tdt_token_ids = np.array(list(y_seq), dtype=np.int32)
    tdt_text = tdt_first.text if hasattr(tdt_first, "text") else str(tdt_first)

    # Tokenizer detok fixture: a small hand-picked set of regular BPE ids
    # (avoid id=0 <unk> and blank_id=1024 which is beyond vocab).
    detok_ids = np.array([10, 25, 100, 3, 7], dtype=np.int32)
    detok_text = m.tokenizer.ids_to_text(detok_ids.tolist())

    # Authoritative CTC transcript of the clip from NeMo's own greedy CTC
    # decoder. parakeet-tdt_ctc-110m is a hybrid; its default transcribe() uses
    # the RNNT/TDT head, so we explicitly switch to the CTC head first. This is
    # the end-to-end ground truth for pk::transcribe (Task 11). transcribe()
    # returns a tuple (list_of_results, optional_beam_results); with
    # return_hypotheses=False each result is a plain text string, but be
    # defensive and extract .text from Hypothesis objects if present.
    m.change_decoding_strategy(decoder_type="ctc")
    assert m.cur_decoder == "ctc", f"expected CTC decoder, got {m.cur_decoder}"
    with torch.no_grad():
        out = m.transcribe([args.audio], batch_size=1, return_hypotheses=False)
    hyps = out[0] if isinstance(out, tuple) else out
    first = hyps[0]
    ctc_text = first.text if hasattr(first, "text") else str(first)

    w = gguf.GGUFWriter(args.output, "parakeet-baseline")
    for k, v in cap.items():
        w.add_tensor(k, _squeeze(v))
    w.add_tensor("ctc_argmax_ids", np.ascontiguousarray(ids))
    w.add_tensor("detok_ids", detok_ids)
    w.add_tensor("pred_input_ids", np.ascontiguousarray(pred_input_ids))
    w.add_tensor("joint_enc_frames", np.ascontiguousarray(joint_enc_frames_arr))
    # tdt_token_ids: NeMo's TDT greedy reference token ids. gguf cannot store a
    # zero-length tensor, so only emit the tensor when non-empty; always record
    # the length as a KV so consumers can distinguish "empty" from "missing".
    w.add_uint32("baseline.tdt_token_count", int(tdt_token_ids.shape[0]))
    if tdt_token_ids.shape[0] > 0:
        w.add_tensor("tdt_token_ids", np.ascontiguousarray(tdt_token_ids))
    w.add_string("baseline.detok_text", detok_text)
    w.add_string("baseline.ctc_text", ctc_text)
    w.add_string("baseline.tdt_text", tdt_text)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    shapes = {k: tuple(_squeeze(v).shape) for k, v in cap.items()}
    shapes["ctc_argmax_ids"] = tuple(ids.shape)
    shapes["detok_ids"] = tuple(detok_ids.shape)
    shapes["pred_input_ids"] = tuple(pred_input_ids.shape)
    shapes["joint_enc_frames"] = tuple(joint_enc_frames_arr.shape)
    if tdt_token_ids.shape[0] > 0:
        shapes["tdt_token_ids"] = tuple(tdt_token_ids.shape)
    print("baseline tensors:", shapes)
    print(f"baseline.detok_text: {repr(detok_text)}")
    print(f"baseline.ctc_text: {repr(ctc_text)}")
    print(f"baseline.tdt_text: {repr(tdt_text)}")
    print(f"tdt_token_ids ({tdt_token_ids.shape[0]}): {tdt_token_ids.tolist()}")
    print(
        f"transducer: pred_input_ids={pred_input_ids.tolist()} add_sos={add_sos} "
        f"U+1={shapes['pred_out'][0]} sos_embed_zero={sos_embed_is_zero} "
        f"joint raw_logits=True joint_enc_frames={n_frames}"
    )
    print(f"wrote {args.output}: tensors={len(shapes)} (dither=0.0, explicit forward)")


if __name__ == "__main__":
    main()
