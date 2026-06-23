import AVFoundation

/// Captures the mic and delivers 16 kHz mono float PCM in small blocks, plus a
/// 0...1 level for the waveform. Uses AVAudioConverter to resample the hardware
/// format (usually 44.1/48 kHz) down to the 16 kHz the model wants.
final class AudioStreamer {
    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private let target = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                       sampleRate: 16000, channels: 1, interleaved: false)!

    /// Called on an audio thread with resampled PCM and the block's RMS level.
    var onPCM: (([Float], Float) -> Void)?

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: [])
        try session.setActive(true)

        let input = engine.inputNode
        let hwFormat = input.outputFormat(forBus: 0)
        converter = AVAudioConverter(from: hwFormat, to: target)

        input.installTap(onBus: 0, bufferSize: 4096, format: hwFormat) { [weak self] buf, _ in
            self?.handle(buf)
        }
        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false)
    }

    private func handle(_ input: AVAudioPCMBuffer) {
        guard let converter else { return }
        // Output capacity scaled by the sample-rate ratio (+slack).
        let ratio = target.sampleRate / input.format.sampleRate
        let cap = AVAudioFrameCount(Double(input.frameLength) * ratio) + 16
        guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: cap) else { return }

        var fed = false
        var err: NSError?
        converter.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return input
        }
        if err != nil || out.frameLength == 0 { return }

        let n = Int(out.frameLength)
        let ptr = out.floatChannelData![0]
        let pcm = Array(UnsafeBufferPointer(start: ptr, count: n))

        var sum: Float = 0
        for v in pcm { sum += v * v }
        let rms = (sum / Float(n)).squareRoot()
        let level = min(1, rms * 6)   // crude visual gain

        onPCM?(pcm, level)
    }
}
