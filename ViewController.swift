import UIKit
import TensorFlowLiteSwift

class ViewController: UIViewController,
                      UIImagePickerControllerDelegate,
                      UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView?
    @IBOutlet weak var resultLabel: UILabel?

    var interpreter: Interpreter?

    override func viewDidLoad() {
        super.viewDidLoad()

        // Load .tflite from app bundle
        guard let modelPath = Bundle.main.path(forResource: "my_multi_out",
                                               ofType: "tflite") else {
            fatalError("Could not find my_multi_out.tflite in the bundle!")
        }
        do {
            interpreter = try Interpreter(modelPath: modelPath)
            try interpreter?.allocateTensors()
            print("TFLite interpreter allocated.")
        } catch {
            print("Error with TFLite interpreter: \(error)")
        }
    }

    @IBAction func pickImageTapped(_ sender: Any) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .photoLibrary
        present(picker, animated: true)
    }

    func imagePickerController(_ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info:
            [UIImagePickerController.InfoKey : Any]) {
        picker.dismiss(animated: true, completion: nil)
        if let uiImage = info[.originalImage] as? UIImage {
            imageView?.image = uiImage
            runInference(uiImage)
        }
    }

    func runInference(_ uiImage: UIImage) {
        guard let interpreter = interpreter else {
            resultLabel?.text = "No interpreter"
            return
        }
        guard let rgbData = uiImage.toRGBData32(width: 64, height: 64) else {
            resultLabel?.text = "Error converting image"
            return
        }

        do {
            try interpreter.copy(rgbData, toInputAt: 0)
            try interpreter.invoke()

            let outAge  = try interpreter.output(at: 0)
            let outGen  = try interpreter.output(at: 1)
            let outExpr = try interpreter.output(at: 2)

            let ageVal = outAge.data.toArray(type: Float.self)[0]
            let genArr = outGen.data.toArray(type: Float.self)
            let exprArr= outExpr.data.toArray(type: Float.self)

            let ageCat = (ageVal>0.5) ? "old" : "young"

            let gIdx = (genArr[0]>genArr[1]) ? 0:1
            let predGender = (gIdx==0) ? "female":"male"

            let eIdx = (exprArr[0]>exprArr[1]) ? 0:1
            let predExpr = (eIdx==0) ? "happy":"sad"

            resultLabel?.text = """
            AgeVal= \(ageVal) => \(ageCat)
            Gender= [\(genArr[0]),\(genArr[1])] => \(predGender)
            Expr= [\(exprArr[0]),\(exprArr[1])] => \(predExpr)
            """
        } catch {
            resultLabel?.text = "Error: \(error)"
        }
    }
}

extension UIImage {
    func toRGBData32(width: Int, height: Int) -> Data? {
        UIGraphicsBeginImageContextWithOptions(CGSize(width: width, height: height), false, 1.0)
        self.draw(in: CGRect(x:0, y:0, width: width, height: height))
        let newImg = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()

        guard let cgImg = newImg?.cgImage else { return nil }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bmpInfo = CGBitmapInfo.byteOrder32Little.rawValue |
                      CGImageAlphaInfo.noneSkipFirst.rawValue
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: 8,
                                      bytesPerRow: width*4,
                                      space: colorSpace,
                                      bitmapInfo: bmpInfo)
        else {
            return nil
        }
        context.draw(cgImg, in: CGRect(x:0, y:0, width: width, height: height))
        guard let buffer = context.data else { return nil }

        return Data(bytes: buffer, count: width*height*4)
    }
}
