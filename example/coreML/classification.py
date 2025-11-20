import os
import numpy as np
import cv2
import onnxruntime as ort
from GUI_Mananger import ExecuteGUI, bmt
from PIL import Image
import torchvision.transforms.functional as F

# Define the interface class for Classification using ONNX
class Classification_Implementation(bmt.AI_BMT_Interface):
     def __init__(self, use_macos_gpu=False, use_customDataset=False):
         super().__init__()
         self.session = None
         self.input_name = None
         self.output_name = None
         self.use_macos_gpu = use_macos_gpu
         self.use_customDataset = use_customDataset
         
         # 추가 (IOBinding 재사용 & 사전할당 버퍼)
         self.io = None
         self.input_shape = None
         self.out_shape = None
         self.out_arr = None

     def getOptionalData(self):
         optional = bmt.Optional_Data()
         optional.cpu_type = "Apple M4 (Python)"
         optional.accelerator_type = "Apple M4 GPU (CoreML)" if self.use_macos_gpu else ""
         optional.submitter = ""
         optional.cpu_core_count = "10"
         optional.cpu_ram_capacity = "24GB"
         optional.cooling = "Passive"
         optional.cooling_option = "Passive"
         optional.cpu_accelerator_interconnect_interface = "Unified Memory" if self.use_macos_gpu else ""
         optional.benchmark_model = ""
         optional.operating_system = "macOS 15.5"
         return optional

     def getInterfaceType(self):
         if self.use_customDataset:
             return bmt.InterfaceType.ImageClassification_CustomDataset
         else:
            return bmt.InterfaceType.ImageClassification

     def initialize(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # ▶ 세션 옵션(최적화/메모리/스레드)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        so.intra_op_num_threads = 0
        so.inter_op_num_threads = 0

        # ▶ EP 고정
        providers = ['CPUExecutionProvider']
        if self.use_macos_gpu:
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']

        # ▶ 세션 생성 (옵션 적용)
        try:
            self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
            if self.use_macos_gpu:
                print("Using CoreML execution provider for GPU acceleration")
        except Exception as e:
            print(f"CoreML EP unavailable, falling back to CPU: {e}")
            self.use_macos_gpu = False
            self.session = ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])

        # ▶ 입/출력 이름
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # ▶ 동적 차원은 1로 고정해 shape 확정
        in_shape  = self.session.get_inputs()[0].shape
        out_shape = self.session.get_outputs()[0].shape
        self.input_shape = [d if isinstance(d, int) and d > 0 else 1 for d in in_shape]   # e.g. [1,3,224,224]
        self.out_shape   = [d if isinstance(d, int) and d > 0 else 1 for d in out_shape]  # e.g. [1,1000]

        # ▶ IOBinding 준비 + 출력 버퍼 사전할당(재사용)
        self.io = ort.IOBinding(self.session)
        self.out_arr = np.empty(self.out_shape, dtype=np.float32)

        # ▶ 워밍업(1회성 변환/최적화 비용 제거)
        dummy = np.zeros(self.input_shape, dtype=np.float32)
        for _ in range(3):
            self.session.run(None, {self.input_name: dummy})

        return True

     def preprocessVisionData(self, image_path: str):
         image = cv2.imread(image_path)
         if image is None:
             raise FileNotFoundError(f"Image not found: {image_path}")
         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

         # Apply custom dataset preprocessing if needed
         if self.use_customDataset:
             image = Image.fromarray(image)
             image = F.resize(image, 232)
             image = F.center_crop(image, [224, 224])
             image = np.array(image)
             
         image = image.astype(np.float32) / 255.0

         # Normalize
         mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
         std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
         image = (image - mean) / std

        # CHW + batch + contiguous 보장
         image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
         image = np.ascontiguousarray(image[None, ...])  # (1,3,224,224)
         return image


     def inferVision(self, preprocessed_data_list):
        output_tensors = []
        for pre in preprocessed_data_list:
            x = np.ascontiguousarray(pre, dtype=np.float32)  # (1,3,224,224)

            in_ptr  = x.ctypes.data
            out_ptr = self.out_arr.ctypes.data

            self.io.bind_input(self.input_name,  'cpu', 0, np.float32, tuple(x.shape),           in_ptr)
            self.io.bind_output(self.output_name, 'cpu', 0, np.float32, tuple(self.out_arr.shape), out_ptr)
            self.session.run_with_iobinding(self.io)

            # 다음 반복 대비 정리
            self.io.clear_binding_inputs()
            self.io.clear_binding_outputs()

            # 재사용 버퍼이므로 copy해서 안전하게 반환
            output_tensors.append(self.out_arr.copy())
        return output_tensors


     def dataTransferVision(self, output_tensors):
         results = []
         for output in output_tensors:
            result = bmt.BMTVisionResult()
            result.classProbabilities = output.flatten()
            results.append(result)
         return results

if __name__ == "__main__":
    # import onnxruntime as ort
    print("onnxRuntime version", ort.__version__)          # 1.23.1
    print("Available providers:", ort.get_available_providers())  # 예: ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    
    # Configuration options
    use_macos_gpu = True  # Set to False for CPU-only mode
    use_customDataset = False
    interface = Classification_Implementation(use_macos_gpu=use_macos_gpu, use_customDataset=use_customDataset)
    print(f"Starting Classification BMT with {'GPU (CoreML)' if use_macos_gpu else 'CPU'} acceleration")
    
    ExecuteGUI(interface)
    
