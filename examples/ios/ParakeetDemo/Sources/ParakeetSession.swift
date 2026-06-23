import Foundation

/// Thin Swift wrapper over the parakeet.cpp flat C streaming API.
/// All calls hop onto one serial queue — the C context is not thread-safe and
/// inference must stay off the main thread.
final class ParakeetSession {
    private var ctx: OpaquePointer?
    private var stream: OpaquePointer?
    private let queue = DispatchQueue(label: "cpp.parakeet.session")

    enum SessionError: Error { case modelMissing, loadFailed(String), beginFailed(String) }

    /// Loads the bundled model.gguf once. Heavy — call off the main thread.
    init() throws {
        guard let path = Bundle.main.path(forResource: "model", ofType: "gguf") else {
            throw SessionError.modelMissing
        }
        guard let c = parakeet_capi_load(path) else {
            throw SessionError.loadFailed("parakeet_capi_load returned null")
        }
        ctx = c
    }

    /// Start a fresh streaming session. `lang` is a locale or "auto".
    func begin(lang: String = "auto") throws {
        try queue.sync {
            if let s = stream { parakeet_capi_stream_free(s); stream = nil }
            guard let s = parakeet_capi_stream_begin_lang(ctx, lang) else {
                let msg = parakeet_capi_last_error(ctx).map { String(cString: $0) } ?? "unknown"
                throw SessionError.beginFailed(msg)
            }
            stream = s
        }
    }

    /// Feed 16 kHz mono float PCM. Returns newly-finalized text (may be empty)
    /// and the EOU/EOB event mask. Runs synchronously on the session queue.
    func feed(_ pcm: [Float]) -> (text: String, eou: Int32) {
        queue.sync {
            guard let s = stream else { return ("", 0) }
            var mask: Int32 = 0
            let out = pcm.withUnsafeBufferPointer {
                parakeet_capi_stream_feed(s, $0.baseAddress, Int32(pcm.count), &mask)
            }
            let text = out.map { String(cString: $0) } ?? ""
            if let out { parakeet_capi_free_string(out) }
            return (text, mask)
        }
    }

    /// Flush the end-of-stream tail.
    func finalize() -> String {
        queue.sync {
            guard let s = stream else { return "" }
            let out = parakeet_capi_stream_finalize(s)
            let text = out.map { String(cString: $0) } ?? ""
            if let out { parakeet_capi_free_string(out) }
            return text
        }
    }

    deinit {
        if let s = stream { parakeet_capi_stream_free(s) }
        if let c = ctx { parakeet_capi_free(c) }
    }
}
