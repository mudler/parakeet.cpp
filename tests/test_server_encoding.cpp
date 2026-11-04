#include "httplib.h"

#include <cstdio>
#include <string>

static int fails = 0;
static void check(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", msg); ++fails; }
}

int main() {
    using namespace httplib;

    // Test has_accept_encoding for "br"
    check(detail::has_accept_encoding("br", "br"), "plain br");
    check(detail::has_accept_encoding("gzip, br", "br"), "gzip, br");
    check(detail::has_accept_encoding("BR", "br"), "case-insensitive BR");
    check(detail::has_accept_encoding("br;q=1.0", "br"), "br;q=1.0");
    check(detail::has_accept_encoding("br;q=0.5", "br"), "br;q=0.5");
    check(detail::has_accept_encoding("gzip;q=0.8, br; q = 0.1", "br"), "br; q = 0.1");

    // Test q=0 rejection for "br"
    check(!detail::has_accept_encoding("br;q=0", "br"), "br;q=0 rejected");
    check(!detail::has_accept_encoding("br;q=0.0", "br"), "br;q=0.0 rejected");
    check(!detail::has_accept_encoding("br;q=0.000", "br"), "br;q=0.000 rejected");
    check(!detail::has_accept_encoding("gzip, br; q = 0", "br"), "br; q = 0 rejected");

    // Test has_accept_encoding for "gzip"
    check(detail::has_accept_encoding("gzip, br; q = 0", "gzip"), "gzip accepted when br;q=0");
    check(!detail::has_accept_encoding("gzip;q=0, br", "gzip"), "gzip;q=0 rejected");
    check(!detail::has_accept_encoding("gzip;q=0.0", "gzip"), "gzip;q=0.0 rejected");

    // Test multiple occurrences and invalid tokens
    check(detail::has_accept_encoding("br;q=0, br;q=1", "br"), "br;q=0, br;q=1 -> accepted");
    check(!detail::has_accept_encoding("zebra", "br"), "zebra != br");
    check(!detail::has_accept_encoding("brotli", "br"), "brotli != br");
    check(!detail::has_accept_encoding("", "br"), "empty -> not accepted");

    // Test encoding_type
    Request req;
    Response res;
    res.set_header("Content-Type", "application/json");

    req.set_header("Accept-Encoding", "br");
    check(detail::encoding_type(req, res) == detail::EncodingType::None,
          "encoding_type returns None when Brotli/Zlib macros not defined");

    if (fails) {
        std::fprintf(stderr, "%d checks failed\n", fails);
        return 1;
    }
    std::printf("test_server_encoding: OK\n");
    return 0;
}
