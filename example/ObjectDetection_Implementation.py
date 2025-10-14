import os
import numpy as np
import cv2
import onnxruntime as ort
from GUI_Mananger import bmt

# Define the interface class for ObjectDetection using ONNX
class ObjectDetection_Implementation(bmt.AI_BMT_Interface):
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
        return bmt.InterfaceType.ObjectDetection
    
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
        
        image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))  # (3, 640, 640)
        return np.array(image, dtype=np.float32).reshape(1, 3, 640, 640)
        
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
            output_tensor = outputs[0] # shape : (1, 25200, 85) for YOLOv5, (1, 84, 8400) for Yolov5u, Yolov8, Yolov9, Yolo11, Yolo12, (1, 300, 6) for Yolov10
            result = bmt.BMTVisionResult()
            result.objectDetectionResult = output_tensor.flatten()
            results.append(result)
        return results
    
    
class ObjectDetection_CustomDataset_Implementation(ObjectDetection_Implementation):
    def getInterfaceType(self):
        return bmt.InterfaceType.ObjectDetection_CustomDataset