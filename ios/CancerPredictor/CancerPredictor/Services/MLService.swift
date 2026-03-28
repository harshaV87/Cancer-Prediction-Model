import CoreML
import Vision
import UIKit

/// Result of a brain tumor classification prediction
struct PredictionResult: Equatable {
    let label: String
    let confidence: Double           // percentage (0-100)
    let inferenceTimeMs: Double      // milliseconds
    let isTumor: Bool

    static func == (lhs: PredictionResult, rhs: PredictionResult) -> Bool {
        lhs.label == rhs.label && lhs.confidence == rhs.confidence
    }
}

/// Service for running CoreML inference on brain MRI images
final class MLService {
    static let shared = MLService()

    private var model: VNCoreMLModel?

    private init() {
        loadModel()
    }

    /// Load the CoreML model
    private func loadModel() {
        // The .mlpackage file must be added to the Xcode project.
        // Xcode auto-generates a Swift class from the .mlpackage name.
        // Replace "BrainTumorClassifier_resnet50_fp16" with your actual model name.
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine + GPU + CPU

            // Try to load the model — the class name matches the .mlpackage filename
            // When you drag the .mlpackage into Xcode, it generates this class automatically.
            // Update the class name below to match your exported .mlpackage filename.
            guard let modelURL = Bundle.main.url(
                forResource: "BrainTumorClassifier_resnet50_fp16",
                withExtension: "mlmodelc"
            ) else {
                print("MLService: Model file not found in bundle.")
                return
            }

            let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
            model = try VNCoreMLModel(for: mlModel)
            print("MLService: Model loaded successfully.")
        } catch {
            print("MLService: Failed to load model — \(error.localizedDescription)")
        }
    }

    /// Run prediction on a UIImage
    /// - Parameter image: The brain MRI image to classify
    /// - Returns: PredictionResult with label, confidence, and timing
    func predict(image: UIImage) async throws -> PredictionResult {
        guard let model = model else {
            throw MLServiceError.modelNotLoaded
        }

        guard let cgImage = image.cgImage else {
            throw MLServiceError.invalidImage
        }

        return try await withCheckedThrowingContinuation { continuation in
            let startTime = CFAbsoluteTimeGetCurrent()

            let request = VNCoreMLRequest(model: model) { request, error in
                let inferenceTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }

                guard let results = request.results as? [VNClassificationObservation],
                      let topResult = results.first else {
                    continuation.resume(throwing: MLServiceError.noPrediction)
                    return
                }

                let result = PredictionResult(
                    label: topResult.identifier,
                    confidence: Double(topResult.confidence) * 100,
                    inferenceTimeMs: inferenceTime,
                    isTumor: topResult.identifier == "Tumor Detected"
                )

                continuation.resume(returning: result)
            }

            // Configure image processing
            request.imageCropAndScaleOption = .scaleFill

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

            do {
                try handler.perform([request])
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

enum MLServiceError: LocalizedError {
    case modelNotLoaded
    case invalidImage
    case noPrediction

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "The ML model could not be loaded."
        case .invalidImage:
            return "The image could not be processed."
        case .noPrediction:
            return "No prediction was returned."
        }
    }
}
