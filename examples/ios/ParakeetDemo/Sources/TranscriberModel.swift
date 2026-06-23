import AVFoundation
import SwiftUI

/// Ties the mic stream to the model and publishes UI state.
@MainActor
final class TranscriberModel: ObservableObject {
    @Published var transcript = ""
    @Published var levels: [Float] = Array(repeating: 0, count: 64)  // waveform ring
    @Published var isRecording = false
    @Published var status = "Tap to start"
    @Published var lang = "en"   // forced language; "auto" tends to mis-detect

    // Display name -> model locale token. Edit freely — an unknown token makes
    // begin() throw (surfaced in `status`).
    let languages: [(name: String, code: String)] = [
        ("Auto-detect", "auto"), ("English", "en"), ("Spanish", "es"),
        ("French", "fr"), ("German", "de"), ("Italian", "it"),
        ("Portuguese", "pt"), ("Dutch", "nl"), ("Russian", "ru"),
        ("Japanese", "ja"), ("Korean", "ko"), ("Chinese", "zh"),
        ("Arabic", "ar"), ("Hindi", "hi"), ("Turkish", "tr"),
    ]

    private let audio = AudioStreamer()
    private var session: ParakeetSession?
    private let work = DispatchQueue(label: "cpp.parakeet.feed")

    func toggle() { isRecording ? stop() : start() }

    private func start() {
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
            Task { @MainActor in
                guard let self else { return }
                guard granted else { self.status = "Microphone denied"; return }
                do {
                    if self.session == nil {
                        self.status = "Loading model…"
                        // Load off the main thread; it's heavy.
                        self.session = try await Task.detached { try ParakeetSession() }.value
                    }
                    try self.session?.begin(lang: self.lang)
                    self.transcript = ""
                    self.audio.onPCM = { [weak self] pcm, level in self?.feed(pcm, level) }
                    // Start the engine off the main thread — AVAudioSession.setActive
                    // on main logs a UI-responsiveness warning.
                    Task.detached {
                        do {
                            try self.audio.start()
                            await MainActor.run { self.isRecording = true; self.status = "Listening…" }
                        } catch {
                            await MainActor.run { self.status = "Error: \(error)" }
                        }
                    }
                } catch {
                    self.status = "Error: \(error)"
                }
            }
        }
    }

    private func stop() {
        audio.stop()
        isRecording = false
        status = "Finishing…"
        work.async { [weak self] in
            let tail = self?.session?.finalize() ?? ""
            Task { @MainActor in
                if !tail.isEmpty { self?.transcript += tail }
                self?.status = "Tap to start"
            }
        }
    }

    // Called on the audio thread.
    private nonisolated func feed(_ pcm: [Float], _ level: Float) {
        Task { @MainActor in self.pushLevel(level) }
        work.async { [weak self] in
            guard let self, let s = self.session else { return }
            let (raw, mask) = s.feed(pcm)
            // Strip inline language tags like "<en-US>" (EOU/EOB are already gone).
            let text = raw.contains("<")
                ? raw.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
                : raw
            guard !text.isEmpty || mask != 0 else { return }
            Task { @MainActor in
                if !text.isEmpty { self.transcript += text }
                if mask & Int32(PARAKEET_EVENT_EOU) != 0 { self.transcript += "\n" }
            }
        }
    }

    private func pushLevel(_ level: Float) {
        levels.removeFirst()
        levels.append(level)
    }
}
