> **Last Updated:** 2025-12-23 (Version 2.1)

---

## 1. Environment

- ISA(Instruction Set Architecture) : ARM64(aarch64)
- OS : macOS 11.0 (Big Sur) or later
- Python Version: **3.8.X ~ 3.13.X supported**

---

## 2. System Requirements

**1. Hardware**

- Apple Silicon Mac (M1, M2, M3, or later)
- 8GB RAM minimum, 16GB recommended

**2. Software Dependencies**

- Python 3.8 or later (conda/homebrew recommended)
- Required Python packages (automatically installed):
  ```bash
  pip install "numpy>=1.21" "opencv-python>=4.5" "onnxruntime>=1.16"
  ```

**3. Hugging Face Authentication (for LLM tasks)**

If you're using Hugging Face models (especially gated models like Llama), you need to authenticate:

1. **Get Hugging Face Token:**

   - Visit https://huggingface.co/settings/tokens
   - Create a new token with "Read" permission
   - Copy the token

2. **Login via CLI:**

   ```bash
   python -m huggingface_hub.commands.huggingface_cli login
   ```

   Enter your token when prompted.

3. **Access Gated Models:**
   - For gated models (e.g., Llama), visit the model page
   - Example: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - Click "Agree and access repository" to accept terms

**Note:** Without authentication, you'll see `hf_token:None` error when loading gated models.

## 3. Project Description

1. Implement AI_BMT_Interface to operate with the intended AI Processing Unit (e.g., CPU, GPU, NPU).
2. Various task example codes are provided. Use these example codes as a reference to implement the interface for the AI Processing Unit.
3. This macOS ARM64 version includes all necessary Qt frameworks and libraries for standalone distribution.

---

## 4. Submitter Development Guide

### Required Interface

submitter **must** subclass `bmt.AI_BMT_Interface` and implement the following methods:

```python
class SubmitterImplementation(bmt.AI_BMT_Interface):

    # Load and initialize your model here
    def initialize(self, model_path: str) -> None:

    # return the implemented interface task type.
    def getInterfaceType(self) -> InterfaceType:

    #  Vision tasks: preprocessing & inference
    #  - preprocessVisionData: convert raw image file into model input format
    #  - inferVision: run inference on preprocessed data and return vision model outputs
    #  - dataTransferVision : transfer vision model outputs to BMT result format
    def preprocessVisionData(self, image_path: str) -> VariantType:
    def inferVision(self, data: List[VariantType]) -> model_outputs:
    def dataTransferVision(self, model_outputs) -> List[BMTVisionResult]:

    # LLM tasks: preprocessing & inference
    # - preprocessLLMData: convert raw input into model input format
    # - inferLLM: run inference on preprocessed data and return LLM model outputs
    # - dataTransferLLM : transfer LLM model outputs to BMT result format
    def preprocessLLMData(self, llmData: LLMPreprocessedInput) -> VariantType:
    def inferLLM(self, data: List[VariantType]) -> model_outputs:
    def dataTransferLLM(self, model_outputs) -> List[BMTLLMResult]:
    
    # LLM MMLU tasks: first token generation for TTFT measurement
    # - inferFirstToken: generate only the first token (AI-BMT will measure the time internally)
    # - Returns void (we only measure TTFT, don't care about the actual first token output)
    # - Only used for MMLU tasks that require TTFT measurement
    def inferFirstToken(self, preprocessed_data)
```

### Optional Interface

submitter can optionally provide hardware/system metadata using:

```python
class SubmitterImplementation(bmt.AI_BMT_Interface):
    def getOptionalData(self) -> Optional_Data:
        data = Optional_Data()
        data.cpu_type = "Apple M1 Pro"  # or M2, M3, etc.
        data.accelerator_type = "Apple Neural Engine"
        data.submitter = "YourCompany"
        data.cpu_core_count = "10"  # 8 performance + 2 efficiency cores
        data.cpu_ram_capacity = "16GB"  # unified memory
        data.cooling = "Air"
        data.cooling_option = "Passive"
        data.cpu_accelerator_interconnect_interface = "Unified Memory Architecture"
        data.benchmark_model = "ResNet-50"
        data.operating_system = "macOS 14.0"  # or your macOS version
        return data
```

## 5. Start BMT

using following commands in `AI_BMT_GUI_Submitter_MacOS_ARM64_Python/` directory.

**Method 1: Direct Python execution**

```bash
python main.py
```
