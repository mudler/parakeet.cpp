#include "openai_format.hpp"
#include "transcription.hpp"

#include <cmath>
#include <cstdio>
#include <string>

static int fails = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); ++fails; }
}
static bool contains(const std::string& h, const std::string& n) {
    return h.find(n) != std::string::npos;
}

int main() {
    using namespace pkserver;

    // parse_format
    Format f;
    check(parse_format("", f) && f == Format::kJson, "empty -> json");
    check(parse_format("json", f) && f == Format::kJson, "json");
    check(parse_format("text", f) && f == Format::kText, "text");
    check(parse_format("verbose_json", f) && f == Format::kVerboseJson, "verbose");
    check(!parse_format("bogus", f), "bogus rejected");

    // json_escape
    check(json_escape("a\"b\\c\n") == "a\\\"b\\\\c\\n", "json_escape basics");

    // Build a transcription
    pk::Transcription tr;
    tr.text = "hello world";
    tr.words = { {"hello", 0.48f, 0.64f, 0.91f}, {"world", 0.66f, 0.85f, 0.89f} };

    Response rt = format_transcription(tr, Format::kText, 7.4, false);
    check(rt.body == "hello world", "text body");
    check(contains(rt.content_type, "text/plain"), "text content-type");

    Response rj = format_transcription(tr, Format::kJson, 7.4, false);
    check(rj.body == "{\"text\":\"hello world\"}", "json body");
    check(contains(rj.content_type, "application/json"), "json content-type");

    Response rv = format_transcription(tr, Format::kVerboseJson, 7.4, true);
    check(contains(rv.body, "\"task\":\"transcribe\""), "verbose task");
    check(contains(rv.body, "\"text\":\"hello world\""), "verbose text");
    check(contains(rv.body, "\"segments\":["), "verbose segments");
    check(contains(rv.body, "\"words\":["), "verbose words present");
    check(contains(rv.body, "\"word\":\"hello\""), "verbose word entry");
    check(contains(rv.body, "\"conf\":0.9100"), "verbose word confidence");

    Response rvn = format_transcription(tr, Format::kVerboseJson, 7.4, false);
    check(!contains(rvn.body, "\"words\":["), "verbose words gated off");

    // A NaN duration must not leak an invalid JSON literal (the fixed() guard
    // emits 0, matching src/transcription_json.cpp's append_json_float).
    Response rnan = format_transcription(tr, Format::kVerboseJson,
                                         std::nan(""), false);
    check(contains(rnan.body, "\"duration\":0"), "NaN duration -> 0");
    check(!contains(rnan.body, "nan"), "no nan literal in json");

    // error envelope
    std::string e = error_body("bad", "invalid_request_error");
    check(contains(e, "\"message\":\"bad\""), "error message");
    check(contains(e, "\"type\":\"invalid_request_error\""), "error type");

    if (fails) { std::fprintf(stderr, "%d checks failed\n", fails); return 1; }
    std::printf("test_server_format: OK\n");
    return 0;
}
