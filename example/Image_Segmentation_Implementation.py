import os
import numpy as np
import cv2
import onnxruntime as ort
from GUI_Mananger import bmt

# Define the interface class for Segmentation using ONNX
class Segmentation_Implementation(bmt.AI_BMT_Interface):
    def __init__(self):
        super().__init__()
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_path = None

    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = ""
        optional.accelerator_type = ""  # e.g., "DeepX M1(NPU)"
        optional.submitter = ""         # e.g., "DeepX"
        optional.cpu_core_count = ""
        optional.cpu_ram_capacity = ""  # e.g., "32GB"
        optional.cooling = ""           # e.g., "Air"
        optional.cooling_option = ""    # e.g., "Active"
        optional.cpu_accelerator_interconnect_interface = ""  # e.g., "PCIe Gen5 x16"
        optional.benchmark_model = ""
        optional.operating_system = ""
        return optional

    def getInterfaceType(self):
        return bmt.InterfaceType.SemanticSegmentation

    def initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        return True

    def preprocessVisionData(self, image_path: str):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB

        # flatten (HWC → 1D)
        image = image.reshape(-1)

        # convert to float and normalize
        image = image.astype(np.float32) / 255.0

        # mean/std normalize per channel (DeepLabv3_mobilenetv2 uses mean/std = [0.5,0.5,0.5])
        if self.model_path.__contains__("v2") or self.model_path.__contains__("V2"):
            means = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            stds = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        else:
            means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
        # Prepare CHW by selecting each channel separately
        chw = []
        for ch in range(3):
            channel_data = image[ch::3]
            normalized = (channel_data - means[ch]) / stds[ch]
            chw.append(normalized)
        return np.array(chw, dtype=np.float32).reshape(1, 3, 520, 520)

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
        output_tensors = []
        for _, preprocessed_data in enumerate(preprocessed_data_list):
            outputs = self.session.run([self.output_name], {self.input_name: preprocessed_data})
            output_tensors.append(outputs[0])
        return output_tensors
    
    def dataTransferVision(self, output_tensors):
        """
        Convert output tensors to BMTVisionResult format.
        This function eliminates wrapper overhead by directly converting
        Python data to C++ compatible format after model inference.
        Do NOT pass multi-dimensional arrays. 
        Use `.flatten()` or `.ravel()` to convert arrays to 1D before assignment.
        """
        results = []
        for output_tensor in output_tensors:
            result = bmt.BMTVisionResult()
            result.segmentationResult = output_tensor.flatten()
            results.append(result)
        return results
    
class Segmentation_CustomDataset_Implementation(Segmentation_Implementation):
    def getInterfaceType(self):
        return bmt.InterfaceType.SemanticSegmentation_CustomDataset