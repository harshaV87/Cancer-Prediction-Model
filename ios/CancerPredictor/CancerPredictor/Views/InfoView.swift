import SwiftUI

struct InfoView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Model info
                GroupBox {
                    VStack(alignment: .leading, spacing: 12) {
                        InfoRow(title: "Model", value: "ResNet-50")
                        InfoRow(title: "Dataset", value: "BraTS 2023 GLI")
                        InfoRow(title: "Input", value: "224 x 224 RGB")
                        InfoRow(title: "Modality", value: "MRI (FLAIR)")
                        InfoRow(title: "Task", value: "Binary Classification")
                        InfoRow(title: "Quantization", value: "Float16")
                        InfoRow(title: "Framework", value: "CoreML (mlprogram)")
                    }
                } label: {
                    Label("Model Details", systemImage: "cpu")
                        .font(.headline)
                }

                // How it works
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("This app uses a deep learning model trained on brain MRI scans from the BraTS 2023 dataset to classify whether an axial MRI slice contains a glioblastoma tumor.")
                        Text("The model runs entirely on-device using Apple's Core ML framework and Neural Engine — no internet connection or server required.")
                    }
                    .font(.subheadline)
                } label: {
                    Label("How It Works", systemImage: "questionmark.circle")
                        .font(.headline)
                }

                // Disclaimer
                GroupBox {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("This application is a research prototype and is NOT intended for clinical diagnosis, treatment decisions, or any medical purpose.")
                            .fontWeight(.semibold)
                        Text("Always consult qualified medical professionals for brain tumor diagnosis and treatment. The predictions are based on a machine learning model and may contain errors.")
                    }
                    .font(.subheadline)
                    .foregroundStyle(.red)
                } label: {
                    Label("Important Disclaimer", systemImage: "exclamationmark.triangle")
                        .font(.headline)
                        .foregroundStyle(.red)
                }
            }
            .padding()
        }
        .navigationTitle("About")
    }
}

struct InfoRow: View {
    let title: String
    let value: String

    var body: some View {
        HStack {
            Text(title)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
        .font(.subheadline)
    }
}

#Preview {
    NavigationStack {
        InfoView()
    }
}
