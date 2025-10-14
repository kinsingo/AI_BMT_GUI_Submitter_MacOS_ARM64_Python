import os
import numpy as np
import cv2
import onnxruntime as ort
from GUI_Mananger import bmt

# Define the interface class for Classification using ONNX
class Classification_Implementation(bmt.AI_BMT_Interface):
    def __init__(self):
        super().__init__()
        self.session = None
        self.input_name = None
        self.output_name = None

    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = ""
        optional.accelerator_type = ""  
        optional.submitter = ""         
        optional.cpu_core_count = ""
        optional.cpu_ram_capacity = ""  # e.g., "32GB"
        optional.cooling = ""           # e.g., "Air"
        optional.cooling_option = ""    # e.g., "Active"
        optional.cpu_accelerator_interconnect_interface = ""  # e.g., "PCIe Gen5 x16"
        optional.benchmark_model = ""
        optional.operating_system = ""
        return optional

    def getInterfaceType(self):
        return bmt.InterfaceType.ImageClassification

    def initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        return True

    def preprocessVisionData(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return np.array(image, dtype=np.float32).reshape(1, 3, 224, 224)

    def inferVision(self, preprocessed_data_list):
        """
        Perform inference and return a list of BMTResult.
        
        Expected: A list of BMTResult objects, each containing one of the following:
        - 'classProbabilities': 1D float list or NumPy array (shape = [1000])
        - 'objectDetectionResult': 1D float list or NumPy array, shape depends on YOLO model:
            - YOLOv5:        [25200 × 85]
            - YOLOv5u/8/9/11/12: [8400 × 84]
            - YOLOv10:       [300 × 6]
        - 'segmentationResult': 1D float list or NumPy array (shape = [21 × 520 × 520])

        Do NOT pass multi-dimensional arrays. 
        Use `.flatten()` or `.ravel()` to convert arrays to 1D before assignment.
        """
        results = []
        for _, preprocessed_data in enumerate(preprocessed_data_list):
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed_data})
            output_tensor = outputs[0]  # shape: (1, 1000)
            result = bmt.BMTVisionResult()
            result.classProbabilities = output_tensor.flatten()
            results.append(result)
        return results
    
    
class Classification_CustomDataset_Implementation(Classification_Implementation):
    def getInterfaceType(self):
        return bmt.InterfaceType.ImageClassification_CustomDataset

    def getResizedAndCenterCroppedImage(self, image):
        # Resize to 232x232 using bilinear interpolation
        image = cv2.resize(image, (232, 232), interpolation=cv2.INTER_LINEAR)
        # Center crop to 224x224
        h, w, _ = image.shape
        top = (h - 224) // 2
        left = (w - 224) // 2
        return image[top:top+224, left:left+224]
        
    def preprocessVisionData(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.getResizedAndCenterCroppedImage(image)
        image = image.astype(np.float32) / 255.0

        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return np.array(image, dtype=np.float32).reshape(1, 3, 224, 224)