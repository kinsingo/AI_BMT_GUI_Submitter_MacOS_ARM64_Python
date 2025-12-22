import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from GUI_Mananger import bmt
from huggingface_hub import HfFolder

# LLM Submitter for Hugging Face (BERT for GLUE)
class LLM_Implementation_HuggingFace_Bert(bmt.AI_BMT_Interface):
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
        return bmt.InterfaceType.LLM_Bert_GLUE

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
            
            # BERT uses SequenceClassification for GLUE tasks
            self.model = AutoModelForSequenceClassification.from_pretrained(
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

    def inferLLM(self, preprocessed_data_list):
        """
        Perform inference with BERT classification model (for GLUE tasks)
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
                
                # token_type_ids can be used with BERT (but not with DistilBERT)
                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "return_dict": True
                }
                
                # Add token_type_ids if present and model supports it
                # DistilBERT, RoBERTa, etc. do not use token_type_ids
                if (len(preprocessed_data.token_type_ids) > 0 and 
                    hasattr(self.model.config, 'type_vocab_size') and 
                    self.model.config.type_vocab_size > 0):
                    token_type_ids = torch.tensor(preprocessed_data.token_type_ids, dtype=torch.long).reshape(
                        preprocessed_data.N, preprocessed_data.S
                    ).to(self.device)
                    kwargs["token_type_ids"] = token_type_ids
                
                # BERT classification model inference
                outputs = self.model(**kwargs)
                
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