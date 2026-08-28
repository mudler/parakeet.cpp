#include "httplib.h"
#include "model_fetch.hpp"
#include "openai_format.hpp"

#include "parakeet.h"        // parakeet_version
#include "model.hpp"          // pk::Model
#include "transcription.hpp"  // pk::Transcription
#include "ggml_graph.hpp"     // pk::set_num_threads, pk::shutdown_backend
#include "dr_wav.h"           // declarations only; impl lives in libparakeet

#include <csignal>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

httplib::Server* g_server = nullptr;
void on_signal(int) { if (g_server) g_server->stop(); }

// Decode WAV bytes in memory into mono float PCM. Returns false if the bytes
// are not a decodable, non-empty WAV (so a bad/empty upload becomes a 400, not
// a 500 from the inference path). The model resamples to 16 kHz internally, so
// the native sample rate is returned as-is.
bool decode_wav_mem(const std::string& bytes, std::vector<float>& mono,
                    int& sample_rate) {
    unsigned int ch = 0, sr = 0;
    drwav_uint64 frames = 0;
    float* pcm = drwav_open_memory_and_read_pcm_frames_f32(
        bytes.data(), bytes.size(), &ch, &sr, &frames, nullptr);
    if (!pcm || ch == 0 || sr == 0 || frames == 0) {
        if (pcm) drwav_free(pcm, nullptr);
        return false;
    }
    mono.resize(frames);
    for (drwav_uint64 i = 0; i < frames; ++i) {
        double acc = 0;
        for (unsigned int c = 0; c < ch; ++c) acc += pcm[i * ch + c];
        mono[i] = (float)(acc / ch);
    }
    drwav_free(pcm, nullptr);
    sample_rate = (int)sr;
    return true;
}

void usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  parakeet-server --model <path|url|alias> [--host 127.0.0.1] "
        "[--port 8080] [--threads N] [--cache-dir <dir>]\n"
        "\n"
        "Serves POST /v1/audio/transcriptions (OpenAI-compatible) for one model.\n");
}

} // namespace

int main(int argc, char** argv) {
    std::string model_arg, host = "127.0.0.1", cache_dir;
    int port = 8080, threads = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model")        model_arg = next("--model");
        else if (a == "--host")    host = next("--host");
        else if (a == "--port")    port = std::atoi(next("--port").c_str());
        else if (a == "--threads") threads = std::atoi(next("--threads").c_str());
        else if (a == "--cache-dir") cache_dir = next("--cache-dir");
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--version" || a == "-V") {
            std::printf("parakeet-server %s\n", parakeet_version());
            return 0;
        }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(); return 2; }
    }
    if (model_arg.empty()) { usage(); return 2; }
    if (cache_dir.empty()) cache_dir = pkserver::default_cache_dir();

    // Resolve and fetch the model before binding the socket.
    pkserver::ModelSource src;
    std::string err;
    if (!pkserver::resolve_model(model_arg, src, err)) {
        std::fprintf(stderr, "parakeet-server: %s\n", err.c_str());
        return 2;
    }
    std::string model_path;
    try {
        if (src.kind == pkserver::ModelSource::kUrl)
            std::fprintf(stderr, "parakeet-server: fetching %s\n", src.value.c_str());
        model_path = pkserver::fetch_model(src, cache_dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "parakeet-server: %s\n", e.what());
        return 1;
    }

    if (threads > 0) pk::set_num_threads(threads);

    std::unique_ptr<pk::Model> model = pk::Model::load(model_path);
    if (!model) {
        std::fprintf(stderr, "parakeet-server: failed to load model %s\n", model_path.c_str());
        pk::shutdown_backend();
        return 1;
    }
    std::mutex infer_mu;

    httplib::Server svr;
    g_server = &svr;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/v1/audio/transcriptions",
             [&](const httplib::Request& req, httplib::Response& res) {
        using namespace pkserver;
        auto fail = [&](int code, const std::string& msg) {
            res.status = code;
            res.set_content(error_body(msg, "invalid_request_error"), "application/json");
        };

        if (!req.has_file("file")) { fail(400, "missing required field 'file'"); return; }

        // response_format (default json)
        Format fmt = Format::kJson;
        if (req.has_file("response_format")) {
            const std::string& rf = req.get_file_value("response_format").content;
            if (!parse_format(rf, fmt)) { fail(400, "unsupported response_format '" + rf + "'"); return; }
        }

        // timestamp_granularities[] = word -> include words[]
        bool include_words = false;
        for (const auto& g : req.get_file_values("timestamp_granularities[]"))
            if (g.content == "word") include_words = true;

        const std::string& bytes = req.get_file_value("file").content;
        std::vector<float> pcm;
        int sr = 0;
        if (!decode_wav_mem(bytes, pcm, sr)) {
            fail(400, "could not decode audio; this example accepts WAV uploads only");
            return;
        }
        double duration_sec = sr > 0 ? (double)pcm.size() / (double)sr : 0.0;

        try {
            pk::Transcription tr;
            {
                std::lock_guard<std::mutex> lock(infer_mu);
                tr = model->transcribe_with_timestamps(pcm, sr, pk::Decoder::kDefault);
            }
            Response out = format_transcription(tr, fmt, duration_sec, include_words);
            res.set_content(out.body, out.content_type.c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "parakeet-server: inference error: %s\n", e.what());
            res.status = 500;
            res.set_content(error_body("internal transcription error", "server_error"),
                            "application/json");
        }
    });

    std::fprintf(stderr, "parakeet-server: listening on http://%s:%d (model: %s)\n",
                 host.c_str(), port, model_path.c_str());
    bool ok = svr.listen(host.c_str(), port);
    g_server = nullptr;
    pk::shutdown_backend();
    if (!ok) {
        std::fprintf(stderr, "parakeet-server: failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
