#include "parakeet.h"
#include "model.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifndef PARAKEET_VERSION
#define PARAKEET_VERSION "0.0.1"
#endif

extern "C" const char* parakeet_version(void) { return PARAKEET_VERSION; }

namespace pk {

// Thin convenience wrapper: load a model and transcribe one file. For repeated
// transcription (and for the flat C-API / LocalAI) use pk::Model directly to
// avoid reloading the GGUF on every call.
std::string transcribe(const std::string& model_path, const std::string& wav_path,
                       Decoder decoder) {
    std::unique_ptr<Model> model = Model::load(model_path);
    if (!model) {
        throw std::runtime_error("parakeet: failed to load model: " + model_path);
    }
    return model->transcribe_path(wav_path, decoder);
}

} // namespace pk

extern "C" int parakeet_transcribe_file(const char* model_path,
                                        const char* wav_path, char** out) {
    if (!model_path || !wav_path || !out) return 1;
    try {
        std::string text = pk::transcribe(model_path, wav_path);
        char* buf = (char*)std::malloc(text.size() + 1);
        if (!buf) return 2;
        std::memcpy(buf, text.c_str(), text.size() + 1);
        *out = buf;
        return 0;
    } catch (const std::exception&) {
        return 3;
    } catch (...) {
        return 4;
    }
}

extern "C" void parakeet_free_string(char* s) { std::free(s); }
