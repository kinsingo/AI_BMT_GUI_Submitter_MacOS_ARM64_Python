import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from GUI_Mananger import bmt
from huggingface_hub import HfFolder

# LLM Submitter for Hugging Face (Decoder-only models)
class LLM_Implementation_HuggingFace_Decoder(bmt.AI_BMT_Interface):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = HfFolder.get_token()  # 현재 로그인된 토큰
        print(f"hf_token:{self.hf_token}")

    def getOptionalData(self):
        optional = bmt.Optional_Data()
        optional.cpu_type = ""
        optional.accelerator_type = ""   # e.g., "DeepX M1(NPU)"
        optional.submitter = ""          # e.g., "DeepX"
        optional.cpu_core_count = ""
        optional.cpu_ram_capacity = ""   # e.g., "32GB"
        optional.cooling = ""            # e.g., "Air"
        optional.cooling_option = ""     # e.g., "Active"
        optional.cpu_accelerator_interconnect_interface = ""  # e.g., "PCIe Gen5 x16"
        optional.benchmark_model = ""
        optional.operating_system = ""
        return optional
    
    def getInterfaceType(self):
        # return bmt.InterfaceType.LLM_QWEN_MMLU
        # return bmt.InterfaceType.LLM_Gemma_MMLU #26.6667
        return bmt.InterfaceType.LLM_Llama_MMLU #50 Ok
        

    def initialize(self, model_path: str):
        try:
            print(f"Loading model from: {model_path}")
            # Load model and tokenizer from Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=self.hf_token,
            )
            print("Tokenizer loaded successfully")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.float32,
                token=self.hf_token,
            )
            print("Model loaded successfully")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"Model moved to device: {self.device}")
            
            # Set pad_token to eos_token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.tokenizer = None
            return False

    def preprocessLLMData(self, llmData: bmt.LLMPreprocessedInput):
        """
        llmData: bmt.LLMPreprocessedInput (fields: input_ids, attention_mask, token_type_ids)
        Preprocessing for Hugging Face model - return as is
        """
        return llmData
    
    def inferFirstToken(self, preprocessed_data):
        """
        MMLU TTFT measurement: generate only the first token
        This method is called by C++ to measure Time To First Token (TTFT).
        The timing is measured in C++ (GlobalTimer), so we just perform inference.
        Returns nothing (void in C++) - we only measure time, not the token itself.
        
        Uses model.generate() with max_new_tokens=1 to actually generate the first new token
        in an autoregressive manner (not just forward pass).
        """
        print("inferFirstToken is called for TTFT measurement..")
        
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please call initialize() first.")
        
        with torch.no_grad():
            # Create input tensors
            input_ids = torch.tensor(preprocessed_data.input_ids, dtype=torch.long).reshape(
                preprocessed_data.N, preprocessed_data.S
            ).to(self.device)
            
            attention_mask = torch.tensor(preprocessed_data.attention_mask, dtype=torch.long).reshape(
                preprocessed_data.N, preprocessed_data.S
            ).to(self.device)
            
            # Generate only the first new token (autoregressive generation)
            # This measures the actual TTFT in autoregressive decoding
            # C++ measures the time, we don't return anything
            _ = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,  # Generate only 1 new token
                do_sample=False,   # Deterministic (greedy decoding)
                pad_token_id=self.tokenizer.pad_token_id
            )
            # Result is discarded - C++ only measures TTFT timing

    def inferLLM(self, preprocessed_data_list):
        """
        Perform inference with Hugging Face model
        """
        print("inferLLM is called..")
        
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please call initialize() first.")
        
        output_tensors = []
        
        with torch.no_grad():
            for preprocessed_data in preprocessed_data_list:
                # Create input tensors
                input_ids = torch.tensor(preprocessed_data.input_ids, dtype=torch.long).reshape(
                    preprocessed_data.N, preprocessed_data.S
                ).to(self.device)
                
                attention_mask = torch.tensor(preprocessed_data.attention_mask, dtype=torch.long).reshape(
                    preprocessed_data.N, preprocessed_data.S
                ).to(self.device)
                
                # Model inference
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Convert logits to numpy
                logits = outputs.logits.cpu().numpy()
                output_tensors.append(logits)
        
        return output_tensors
    
    def dataTransferLLM(self, output_tensors):
        """
        Convert output tensors to BMTLLMResult format.
        This function eliminates wrapper overhead by directly converting
        Python data to C++ compatible format after model inference.
        Do NOT pass multi-dimensional arrays. 
        Use `.flatten()` or `.ravel()` to convert arrays to 1D before assignment.
        """
        results = []
        for output_tensor in output_tensors:
            r = bmt.BMTLLMResult()
            r.rawOutputShape = [int(x) for x in output_tensor.shape]
            r.rawOutput = np.asarray(output_tensor, dtype=np.float32).ravel().tolist()
            results.append(r)
        return results