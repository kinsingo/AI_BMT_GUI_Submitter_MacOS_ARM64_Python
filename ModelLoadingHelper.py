from torch import nn
import torch.nn.functional as F

class DeeplabWithUpsample(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        out = self.model(pixel_values=x)
        logits = out.logits
        logits_up = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits_up}