import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var viewModel = PredictionViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                // Header
                VStack(spacing: 8) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 60))
                        .foregroundStyle(.blue)
                    Text("Brain Tumor Classifier")
                        .font(.title.bold())
                    Text("MRI Slice Analysis")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 20)

                Spacer()

                // Image preview
                if let image = viewModel.selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 250)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .shadow(radius: 4)
                } else {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.gray.opacity(0.1))
                        .frame(height: 250)
                        .overlay {
                            VStack(spacing: 8) {
                                Image(systemName: "photo.on.rectangle.angled")
                                    .font(.largeTitle)
                                    .foregroundStyle(.secondary)
                                Text("Select an MRI image")
                                    .foregroundStyle(.secondary)
                            }
                        }
                }

                // Result display
                if let result = viewModel.predictionResult {
                    ResultCard(result: result)
                        .transition(.scale.combined(with: .opacity))
                }

                Spacer()

                // Action buttons
                VStack(spacing: 12) {
                    PhotosPicker(
                        selection: $viewModel.photoSelection,
                        matching: .images
                    ) {
                        Label("Select from Photos", systemImage: "photo.on.rectangle")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.blue)
                            .foregroundStyle(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }

                    Button {
                        viewModel.showCamera = true
                    } label: {
                        Label("Take Photo", systemImage: "camera")
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.blue.opacity(0.1))
                            .foregroundStyle(.blue)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                }

                // Disclaimer
                Text("For research purposes only. Not for clinical diagnosis.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.bottom, 8)
            }
            .padding(.horizontal)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink {
                        InfoView()
                    } label: {
                        Image(systemName: "info.circle")
                    }
                }
            }
            .sheet(isPresented: $viewModel.showCamera) {
                CameraView(image: $viewModel.cameraImage)
            }
            .animation(.easeInOut, value: viewModel.predictionResult != nil)
        }
    }
}

// MARK: - Result Card

struct ResultCard: View {
    let result: PredictionResult

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Image(systemName: result.isTumor ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                    .font(.title2)
                    .foregroundStyle(result.isTumor ? .red : .green)

                Text(result.label)
                    .font(.title3.bold())
                    .foregroundStyle(result.isTumor ? .red : .green)
            }

            HStack(spacing: 24) {
                VStack {
                    Text("Confidence")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(result.confidence, specifier: "%.1f")%")
                        .font(.title2.bold())
                }

                VStack {
                    Text("Inference")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(result.inferenceTimeMs, specifier: "%.0f") ms")
                        .font(.title2.bold())
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(result.isTumor ? Color.red.opacity(0.08) : Color.green.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(result.isTumor ? Color.red.opacity(0.3) : Color.green.opacity(0.3), lineWidth: 1)
        )
    }
}

#Preview {
    ContentView()
}
