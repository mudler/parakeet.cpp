function(pk_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE parakeet)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

pk_add_test(test_smoke)
pk_add_test(test_audio_io)
pk_add_test(test_model_loader)
pk_add_test(test_fft)
pk_add_test(test_mel)
pk_add_test(test_mel_gpu)
pk_add_test(test_subsampling)
pk_add_test(test_subsampling_batch)
pk_add_test(test_relpos_attention)
pk_add_test(test_relpos_attention_local)
pk_add_test(test_relpos_attention_local_chunked)
pk_add_test(test_relpos_attention_local_memory)
pk_add_test(test_relpos_attention_batch)
pk_add_test(test_conformer)
pk_add_test(test_conformer_batch)
pk_add_test(test_conv_eou)
pk_add_test(test_encoder)
pk_add_test(test_encoder_batch)
pk_add_test(test_encoder_batch_local)
pk_add_test(test_encoder_eou)
pk_add_test(test_streaming_encoder)
pk_add_test(test_ctc)
pk_add_test(test_prediction)
pk_add_test(test_prediction_step)
pk_add_test(test_prediction_step_batch)
pk_add_test(test_joint)
pk_add_test(test_joint_step_batch)
pk_add_test(test_prompt_kernel)
pk_add_test(test_transducer_core)
pk_add_test(test_tdt_greedy)
pk_add_test(test_transducer_greedy_batch)
pk_add_test(test_transducer_greedy_batch_rnnt)
pk_add_test(test_timestamps_tokens)
pk_add_test(test_timestamps)
pk_add_test(test_transcribe_batch_ts)
pk_add_test(test_tokenizer)
pk_add_test(test_transcribe)
pk_add_test(test_transcribe_speech)
pk_add_test(test_transcribe_tdt)
pk_add_test(test_transcribe_0_6b)
pk_add_test(test_transcribe_ctc)
pk_add_test(test_transcribe_rnnt)
pk_add_test(test_transcribe_eou)
pk_add_test(test_transcribe_nemotron)
pk_add_test(test_streaming_decode)
pk_add_test(test_streaming_nemotron)
pk_add_test(test_streaming_mel)
pk_add_test(test_capi)
pk_add_test(test_capi_batch)
pk_add_test(test_capi_stream)
pk_add_test(test_capi_timestamps)
pk_add_test(test_capi_batch_json)
set_tests_properties(test_model_loader test_mel test_mel_gpu test_subsampling test_subsampling_batch test_relpos_attention test_relpos_attention_batch test_relpos_attention_local_chunked
                     test_conformer test_conformer_batch test_conv_eou test_encoder test_encoder_batch test_encoder_batch_local test_encoder_eou
                     test_streaming_encoder test_ctc test_prediction
                     test_prediction_step test_prediction_step_batch
                     test_joint test_joint_step_batch test_prompt_kernel test_transducer_core test_tdt_greedy
                     test_transducer_greedy_batch test_transducer_greedy_batch_rnnt
                     test_timestamps_tokens test_timestamps test_transcribe_batch_ts test_tokenizer test_transcribe
                     test_transcribe_speech test_transcribe_tdt test_transcribe_0_6b
                     test_transcribe_ctc test_transcribe_rnnt test_transcribe_eou test_transcribe_nemotron
                     test_streaming_decode test_streaming_nemotron test_streaming_mel test_capi test_capi_batch test_capi_stream
                     test_capi_timestamps test_capi_batch_json
                     PROPERTIES LABELS "model")
# These tests read fixtures/baselines via paths relative to the project root.
set_tests_properties(test_mel test_mel_gpu test_subsampling test_subsampling_batch test_relpos_attention test_relpos_attention_batch test_conformer test_conformer_batch
                     test_conv_eou test_encoder test_encoder_batch test_encoder_batch_local test_encoder_eou test_streaming_encoder
                     test_ctc test_prediction test_prediction_step test_prediction_step_batch
                     test_joint test_joint_step_batch test_prompt_kernel
                     test_transducer_core test_tdt_greedy
                     test_transducer_greedy_batch test_transducer_greedy_batch_rnnt
                     test_timestamps_tokens
                     test_timestamps test_transcribe_batch_ts
                     test_tokenizer test_transcribe
                     test_transcribe_speech test_transcribe_tdt test_transcribe_0_6b
                     test_transcribe_ctc test_transcribe_rnnt test_transcribe_eou test_transcribe_nemotron
                     test_streaming_decode test_streaming_nemotron test_streaming_mel test_capi test_capi_batch test_capi_stream
                     test_capi_timestamps test_capi_batch_json
                     PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Python converter check (skips with exit 77 when the venv/model are absent).
find_program(PARAKEET_PY NAMES python3 HINTS ${CMAKE_SOURCE_DIR}/.venv/bin)
if(PARAKEET_PY)
  add_test(NAME check_convert COMMAND ${PARAKEET_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_convert.py)
  set_tests_properties(check_convert PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")

  add_test(NAME check_baseline COMMAND ${PARAKEET_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_baseline.py)
  set_tests_properties(check_baseline PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
endif()
