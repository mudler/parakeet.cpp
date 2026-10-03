#pragma once
// Bundle GGUF: several single-model GGUFs in one file (format in docs/bundle.md).
//
// A bundle has general.architecture "parakeet-bundle". Every tensor and key of
// a component is stored as "<component>.<original name>". The header lists the
// components and carries the licence and source of each one. Reading the header
// never touches tensor data.
#include <cstdint>
#include <string>
#include <vector>

namespace pk {

constexpr const char* kBundleArch = "parakeet-bundle";
constexpr uint32_t kBundleVersion = 1;   // highest format version this reader knows

// Component kinds understood in phase 1. A bundle may hold other kinds (for
// example from a newer writer); a reader skips them.
constexpr const char* kBundleKindAsr = "asr";
constexpr const char* kBundleKindVad = "vad";   // Silero VAD

struct BundleComponent {
    std::string name;
    std::string kind;
    std::string license;
    std::string license_url;
    std::string source;
    std::string attribution;
    std::string changes;
    std::string source_sha256;    // sha256 of the single-model GGUF it was built from
    std::string content_sha256;   // sha256 over the component's tensors (see docs/bundle.md)
    uint64_t n_tensors = 0;
    uint64_t n_bytes = 0;         // tensor data bytes
};

struct BundleInfo {
    uint32_t version = 0;
    std::string name;
    std::vector<BundleComponent> components;
    const BundleComponent* find(const std::string& name) const;
};

// True when the file is a GGUF whose general.architecture is "parakeet-bundle".
// Reads the header only.
bool gguf_is_bundle(const std::string& path);

// Reads and checks the bundle header (no tensor data). Returns false and writes a
// one-line reason to *err on a file that is not a bundle, has a newer format
// version, or has a malformed component table.
bool read_bundle_info(const std::string& path, BundleInfo& out, std::string* err);

// Picks the component that `parakeet_capi_load` opens for a bundle: the only
// component of kind "asr"; failing that, the only component of any known kind.
// Returns false with a message that names the choices when there is none or
// when it is ambiguous.
bool select_default_component(const BundleInfo& info, std::string& name, std::string* err);

// JSON array: [{"name","kind","license","license_url","source","attribution",
// "changes","source_sha256","content_sha256","tensors","bytes"}]
std::string bundle_components_json(const BundleInfo& info);

// "<name> (<kind>), ..." for messages.
std::string bundle_component_names(const BundleInfo& info);

}  // namespace pk
