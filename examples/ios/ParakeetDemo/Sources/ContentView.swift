import SwiftUI

struct ContentView: View {
    @StateObject private var model = TranscriberModel()

    var body: some View {
        VStack(spacing: 24) {
            // Top: live audio waveform.
            Waveform(levels: model.levels)
                .frame(height: 120)
                .padding(.horizontal)

            // Middle: start / stop.
            Button(action: model.toggle) {
                Image(systemName: model.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                    .resizable()
                    .frame(width: 84, height: 84)
                    .foregroundStyle(model.isRecording ? .red : .accentColor)
            }
            Text(model.status).font(.footnote).foregroundStyle(.secondary)

            // Language — disabled while recording (applies on next start).
            Picker("Language", selection: $model.lang) {
                ForEach(model.languages, id: \.code) { Text($0.name).tag($0.code) }
            }
            .pickerStyle(.menu)
            .disabled(model.isRecording)

            // Bottom: live transcription.
            ScrollView {
                Text(model.transcript.isEmpty ? "…" : model.transcript)
                    .font(.title3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal)
        }
        .padding(.vertical)
    }
}

/// Minimal bar waveform driven by the level ring buffer. Native Canvas, no deps.
struct Waveform: View {
    let levels: [Float]
    var body: some View {
        Canvas { ctx, size in
            let n = levels.count
            guard n > 0 else { return }
            let w = size.width / CGFloat(n)
            for (i, lvl) in levels.enumerated() {
                let h = max(2, CGFloat(lvl) * size.height)
                let x = CGFloat(i) * w
                let rect = CGRect(x: x + 1, y: (size.height - h) / 2, width: w - 2, height: h)
                ctx.fill(Path(roundedRect: rect, cornerRadius: 1), with: .color(.accentColor))
            }
        }
    }
}
