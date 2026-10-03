#include "bundle.hpp"

#include <cstdio>
#include <set>

#include "ggml.h"
#include "gguf.h"

namespace pk {

namespace {

void set_err(std::string* err, const std::string& m) {
    if (err) *err = m;
}

struct Gguf {
    gguf_context* g = nullptr;
    ~Gguf() {
        if (g) gguf_free(g);
    }
};

bool get_str(gguf_context* g, const std::string& key, std::string& out) {
    const int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) return false;
    out = gguf_get_val_str(g, id);
    return true;
}

bool valid_component_name(const std::string& n) {
    if (n.empty() || n.size() > 32 || !(n[0] >= 'a' && n[0] <= 'z')) return false;
    for (char c : n)
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_')) return false;
    return n != "general" && n != "parakeet" && n != "bundle";
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", c);
                    o += b;
                } else {
                    o += (char)c;
                }
        }
    }
    return o;
}

}  // namespace

const BundleComponent* BundleInfo::find(const std::string& n) const {
    for (const auto& c : components)
        if (c.name == n) return &c;
    return nullptr;
}

bool gguf_is_bundle(const std::string& path) {
    gguf_init_params ip{/*no_alloc=*/true, /*ctx=*/nullptr};
    Gguf f;
    f.g = gguf_init_from_file(path.c_str(), ip);
    if (!f.g) return false;
    std::string arch;
    return get_str(f.g, "general.architecture", arch) && arch == kBundleArch;
}

bool read_bundle_info(const std::string& path, BundleInfo& out, std::string* err) {
    out = BundleInfo();
    gguf_init_params ip{/*no_alloc=*/true, /*ctx=*/nullptr};
    Gguf f;
    f.g = gguf_init_from_file(path.c_str(), ip);
    if (!f.g) {
        set_err(err, "cannot read " + path + " as a GGUF file");
        return false;
    }
    gguf_context* g = f.g;
    std::string arch;
    if (!get_str(g, "general.architecture", arch) || arch != kBundleArch) {
        set_err(err, path + " is not a bundle (general.architecture is not \"" + kBundleArch + "\")");
        return false;
    }
    const int64_t vid = gguf_find_key(g, "parakeet.bundle.version");
    if (vid < 0 || gguf_get_kv_type(g, vid) != GGUF_TYPE_UINT32) {
        set_err(err, "bundle has no parakeet.bundle.version");
        return false;
    }
    out.version = gguf_get_val_u32(g, vid);
    if (out.version == 0 || out.version > kBundleVersion) {
        set_err(err, "bundle format version " + std::to_string(out.version) +
                         " is not supported by this reader (it knows up to " + std::to_string(kBundleVersion) +
                         "); update parakeet.cpp");
        return false;
    }
    get_str(g, "general.name", out.name);
    const int64_t cid = gguf_find_key(g, "parakeet.bundle.components");
    if (cid < 0 || gguf_get_kv_type(g, cid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, cid) != GGUF_TYPE_STRING) {
        set_err(err, "bundle has no parakeet.bundle.components string array");
        return false;
    }
    const size_t n = gguf_get_arr_n(g, cid);
    if (n == 0) {
        set_err(err, "bundle lists no components");
        return false;
    }
    std::set<std::string> seen;
    for (size_t i = 0; i < n; ++i) {
        BundleComponent c;
        c.name = gguf_get_arr_str(g, cid, i);
        if (!valid_component_name(c.name)) {
            set_err(err, "bundle component name \"" + c.name + "\" is not valid");
            return false;
        }
        if (!seen.insert(c.name).second) {
            set_err(err, "bundle lists component \"" + c.name + "\" twice");
            return false;
        }
        const std::string pre = "parakeet.bundle." + c.name + ".";
        if (!get_str(g, pre + "kind", c.kind) || c.kind.empty()) {
            set_err(err, "component \"" + c.name + "\" has no " + pre + "kind");
            return false;
        }
        get_str(g, pre + "license", c.license);
        get_str(g, pre + "license_url", c.license_url);
        get_str(g, pre + "source", c.source);
        get_str(g, pre + "attribution", c.attribution);
        get_str(g, pre + "changes", c.changes);
        get_str(g, pre + "source_sha256", c.source_sha256);
        get_str(g, pre + "content_sha256", c.content_sha256);
        out.components.push_back(std::move(c));
    }
    const int64_t nt = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < nt; ++i) {
        const std::string tn = gguf_get_tensor_name(g, i);
        const size_t dot = tn.find('.');
        if (dot == std::string::npos) continue;
        for (auto& c : out.components) {
            if (tn.compare(0, dot, c.name) == 0 && dot == c.name.size()) {
                ++c.n_tensors;
                c.n_bytes += gguf_get_tensor_size(g, i);
                break;
            }
        }
    }
    for (const auto& c : out.components) {
        if (c.n_tensors == 0) {
            set_err(err, "component \"" + c.name + "\" has no tensors");
            return false;
        }
    }
    return true;
}

std::string bundle_component_names(const BundleInfo& info) {
    std::string s;
    for (const auto& c : info.components) s += (s.empty() ? "" : ", ") + c.name + " (" + c.kind + ")";
    return s;
}

bool select_default_component(const BundleInfo& info, std::string& name, std::string* err) {
    std::vector<const BundleComponent*> asr, known;
    for (const auto& c : info.components) {
        if (c.kind == kBundleKindAsr) asr.push_back(&c);
        if (c.kind == kBundleKindAsr || c.kind == kBundleKindVad) known.push_back(&c);
    }
    const std::vector<const BundleComponent*>& pick = !asr.empty() ? asr : known;
    if (pick.size() == 1) {
        name = pick[0]->name;
        return true;
    }
    if (pick.empty())
        set_err(err, "bundle has no component this reader can load; components: " + bundle_component_names(info));
    else
        set_err(err, std::string("bundle has several ") + (asr.empty() ? "loadable" : "asr") +
                         " components, pick one by name (parakeet_capi_load_component, or --component); "
                         "components: " + bundle_component_names(info));
    return false;
}

std::string bundle_components_json(const BundleInfo& info) {
    std::string o = "[";
    for (size_t i = 0; i < info.components.size(); ++i) {
        const auto& c = info.components[i];
        if (i) o += ",";
        o += "{\"name\":\"" + json_escape(c.name) + "\",\"kind\":\"" + json_escape(c.kind) +
             "\",\"license\":\"" + json_escape(c.license) + "\",\"license_url\":\"" + json_escape(c.license_url) +
             "\",\"source\":\"" + json_escape(c.source) + "\",\"attribution\":\"" + json_escape(c.attribution) +
             "\",\"changes\":\"" + json_escape(c.changes) + "\",\"source_sha256\":\"" + json_escape(c.source_sha256) +
             "\",\"content_sha256\":\"" + json_escape(c.content_sha256) + "\",\"tensors\":" +
             std::to_string(c.n_tensors) + ",\"bytes\":" + std::to_string(c.n_bytes) + "}";
    }
    return o + "]";
}

}  // namespace pk
