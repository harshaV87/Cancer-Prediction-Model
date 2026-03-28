import SwiftUI
import PhotosUI

@MainActor
final class PredictionViewModel: ObservableObject {
    @Published var selectedImage: UIImage?
    @Published var predictionResult: PredictionResult?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var showCamera = false

    @Published var photoSelection: PhotosPickerItem? {
        didSet {
            if let photoSelection {
                loadImage(from: photoSelection)
            }
        }
    }

    var cameraImage: UIImage? {
        didSet {
            if let cameraImage {
                selectedImage = cameraImage
                runPrediction(on: cameraImage)
            }
        }
    }

    private let mlService = MLService.shared

    private func loadImage(from item: PhotosPickerItem) {
        Task {
            do {
                guard let data = try await item.loadTransferable(type: Data.self),
                      let image = UIImage(data: data) else {
                    errorMessage = "Failed to load image."
                    return
                }
                selectedImage = image
                runPrediction(on: image)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func runPrediction(on image: UIImage) {
        Task {
            isLoading = true
            predictionResult = nil
            errorMessage = nil

            do {
                let result = try await mlService.predict(image: image)
                predictionResult = result
            } catch {
                errorMessage = error.localizedDescription
            }

            isLoading = false
        }
    }
}
