cmake_minimum_required(VERSION 3.18)
project(parakeet_cpp LANGUAGES C CXX)

set(PARAKEET_VERSION "0.0.1" CACHE STRING "Version string returned by parakeet_version()")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PARAKEET_BUILD_TESTS "Build ctest targets" OFF)
option(PARAKEET_BUILD_CLI   "Build parakeet-cli" ON)
option(PARAKEET_BUILD_SERVER "Build parakeet-server (OpenAI-compatible example)" ON)
option(PARAKEET_SHARED      "Build libparakeet as shared" OFF)
option(PARAKEET_GGML_CUDA   "Forward GGML_CUDA" OFF)
option(PARAKEET_GGML_METAL  "Forward GGML_METAL" OFF)
option(PARAKEET_GGML_VULKAN "Forward GGML_VULKAN" OFF)
option(PARAKEET_GGML_HIP "Forward GGML_HIP (ROCm)" OFF)
option(PARAKEET_WITH_CED "Sound-event detection through ced.cpp" ON)
option(PARAKEET_WITH_VOICEDETECT "Speaker identification through voice-detect.cpp" ON)

set(GGML_CUDA    ${PARAKEET_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_METAL   ${PARAKEET_GGML_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN  ${PARAKEET_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_HIP ${PARAKEET_GGML_HIP} CACHE BOOL "" FORCE)

# CUDA graphs: ggml defaults this OFF, but on CUDA builds it captures/replays the
# compute graph for a small free speedup (~1% measured on GB10, never negative).
# Turn it on whenever we forward CUDA so every CUDA build (docker/release/local)
# gets it. The runtime kill-switch GGML_CUDA_DISABLE_GRAPHS=1 still works.
if(PARAKEET_GGML_CUDA)
  set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()

# Performance: -march=native (GGML_NATIVE) and tinyBLAS SGEMM (GGML_LLAMAFILE)
# give meaningful free speedups (~30% and ~25% per the rt-detr.cpp benchmarks).
# Force them on unless the caller explicitly overrides.
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "ggml: optimize the build for the current system" FORCE)
endif()
if(NOT DEFINED GGML_LLAMAFILE)
  set(GGML_LLAMAFILE ON CACHE BOOL "ggml: use LLAMAFILE (tinyBLAS SGEMM)" FORCE)
endif()

# Apply any in-tree ggml patches (broadcast-fold, etc.) before configuring the
# submodule so the patched source is compiled in. Idempotent: already-applied
# patches are silently skipped.
find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh")
  execute_process(
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _patch_result
    OUTPUT_VARIABLE _patch_output
    ERROR_VARIABLE  _patch_errors
  )
  if(_patch_output)
    message(STATUS "${_patch_output}")
  endif()
  if(_patch_result)
    message(WARNING "apply_ggml_patches.sh returned ${_patch_result}: ${_patch_errors}")
  endif()
endif()

add_subdirectory(third_party/ggml)

# Single dr_wav implementation for the whole build. src/audio_io.cpp only
# declares dr_wav's functions (#include "dr_wav.h", no DR_WAV_IMPLEMENTATION);
# this object library is the one translation unit that defines them, so both
# libparakeet and, when PARAKEET_WITH_CED is on, ced.cpp (built with
# CED_EXTERNAL_DR_WAV, which likewise only declares them) link against the
# same symbols instead of each pulling in its own copy (a second
# DR_WAV_IMPLEMENTATION define would be a multiple-definition link error).
# An OBJECT library's objects are linked directly into each consumer rather
# than referenced as a separate archive dependency, so this does not create a
# link-time cycle between the parakeet and ced targets.
add_library(dr_wav_impl OBJECT src/dr_wav_impl.cpp)
target_include_directories(dr_wav_impl PRIVATE ${CMAKE_SOURCE_DIR}/third_party)
set_target_properties(dr_wav_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(PARAKEET_WITH_CED)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ced.cpp/CMakeLists.txt")
    message(FATAL_ERROR "third_party/ced.cpp is missing: run `git submodule update --init --recursive`, or configure with -DPARAKEET_WITH_CED=OFF")
  endif()
  set(CED_BUILD_CLI OFF CACHE BOOL "" FORCE)
  set(CED_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(CED_SHARED OFF CACHE BOOL "" FORCE)
  set(CED_EXTERNAL_DR_WAV ON CACHE BOOL "" FORCE)   # dr_wav_impl provides it
  add_subdirectory(third_party/ced.cpp EXCLUDE_FROM_ALL)
  set_target_properties(ced PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(ced PRIVATE dr_wav_impl)
endif()

if(PARAKEET_WITH_VOICEDETECT)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/voice-detect.cpp/CMakeLists.txt")
    message(FATAL_ERROR "third_party/voice-detect.cpp is missing: run `git submodule update --init third_party/voice-detect.cpp`, or configure with -DPARAKEET_WITH_VOICEDETECT=OFF")
  endif()
  set(VOICEDETECT_BUILD_CLI OFF CACHE BOOL "" FORCE)
  set(VOICEDETECT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(VOICEDETECT_SHARED OFF CACHE BOOL "" FORCE)
  set(VOICEDETECT_EXTERNAL_DR_WAV ON CACHE BOOL "" FORCE)   # dr_wav_impl provides it
  add_subdirectory(third_party/voice-detect.cpp EXCLUDE_FROM_ALL)
  set_target_properties(voicedetect PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(voicedetect PRIVATE dr_wav_impl)
endif()

set(PARAKEET_SRC
    src/parakeet.cpp
    src/model.cpp
    src/vad_head.cpp
    src/vad_segmenter.cpp
    src/vad_json.cpp
    src/bundle.cpp
    src/parakeet_capi.cpp
    src/common.cpp
    src/audio_io.cpp
    src/transcription_json.cpp
    src/model_loader.cpp
    src/backend.cpp
    src/ggml_graph.cpp
    src/backend_pool.cpp
    src/fft.cpp
    src/mel.cpp
    src/mel_gpu.cpp
    src/subsampling.cpp
    src/relpos_attention.cpp
    src/conformer.cpp
    src/pos_enc.cpp
    src/encoder.cpp
    src/streaming_encoder.cpp
    src/streaming.cpp
    src/ctc_decoder.cpp
    src/prediction.cpp
    src/joint.cpp
    src/prompt_kernel.cpp
    src/ternary.cpp
    src/exact_matvec.cpp
    src/ternary_kernels_x86.cpp
    src/ternary_kernels_neon.cpp
    src/tdt.cpp
    src/rnnt.cpp
    src/transducer_batch.cpp
    src/tokenizer.cpp
    src/search.cpp
    src/transcription.cpp
    src/diarization.cpp
    src/diarization_encoder.cpp
    src/diarization_head.cpp
    src/sas_merge.cpp
    src/asr_committer.cpp
    src/diar_pcm_stream.cpp
    src/scene_stream.cpp
    src/diarization_streaming.cpp
    src/ced_tagger.cpp
    src/sound_stream.cpp
    src/speaker_registry.cpp
    src/speaker_identifier.cpp
    src/speaker_encoder.cpp
    src/scene_render.cpp
    src/silero_vad.cpp)

if(PARAKEET_SHARED)
  add_library(parakeet SHARED ${PARAKEET_SRC})
  # The C-API header marks symbols with plain extern "C", no __declspec(dllexport).
  # On MSVC that yields a DLL with no exported symbols (and no import .lib), which
  # is useless for dlopen/ctypes consumers. Auto-export every symbol so the DLL is
  # usable the same way the ELF/Mach-O shared libs are. No effect on Linux/macOS.
  set_target_properties(parakeet PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(parakeet STATIC ${PARAKEET_SRC})
endif()
if(NOT MSVC)
  set_source_files_properties(src/ternary.cpp src/ternary_kernels_x86.cpp src/ternary_kernels_neon.cpp
                              PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND NOT APPLE AND NOT MSVC)
  set_property(SOURCE src/ternary_kernels_neon.cpp APPEND PROPERTY COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
endif()
target_include_directories(parakeet PUBLIC include PRIVATE src ${CMAKE_SOURCE_DIR}/third_party)
target_compile_definitions(parakeet PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
target_compile_definitions(parakeet PRIVATE PARAKEET_VERSION="${PARAKEET_VERSION}")
target_link_libraries(parakeet PUBLIC ggml)
target_link_libraries(parakeet PRIVATE dr_wav_impl)

if(PARAKEET_WITH_CED)
  target_link_libraries(parakeet PRIVATE ced)
  target_compile_definitions(parakeet PRIVATE PARAKEET_WITH_CED=1)
endif()

if(PARAKEET_WITH_VOICEDETECT)
  target_link_libraries(parakeet PRIVATE voicedetect)
  target_compile_definitions(parakeet PRIVATE PARAKEET_WITH_VOICEDETECT=1)
endif()

if(PARAKEET_BUILD_CLI)
  add_subdirectory(examples/cli)
endif()
if(PARAKEET_BUILD_SERVER)
  add_subdirectory(examples/server)
endif()
if(PARAKEET_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
