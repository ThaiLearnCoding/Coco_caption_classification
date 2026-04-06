import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class CLIPZeroShotClassifier(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

    def forward(self, image, text_candidates):
        # image shape (B, C, H, W)
        # text_candidates shape tuple of length K, each containing B strings
        B = image.shape[0]
        K = len(text_candidates)
        
        # Tokenize text
        # Flatten the list of lists into a single list
        flat_texts = [text for sublist in text_candidates for text in sublist]
        text_tokens = clip.tokenize(flat_texts).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_tokens)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Reshape text features back to (B, K, C)
            C = image_features.shape[-1]
            text_features = text_features.view(K, B, C).transpose(0, 1)

            # Compute similarities: (B, 1, C) @ (B, C, K) -> (B, 1, K) -> (B, K)
            logit_scale = self.model.logit_scale.exp()
            
            # Using torch.bmm for batched matrix multiplication
            # image_features: (B, C) -> (B, 1, C)
            # text_features: (B, C, K) -> text_features.transpose(1, 2)
            similarities = logit_scale * torch.bmm(
                image_features.unsqueeze(1),
                text_features.transpose(1, 2)
            ).squeeze(1)

        return similarities

class ResidualDualProbeClassifier(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu", n_captions=12):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        
        # Freeze CLIP backbone
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        
        # Extract features dim based on model
        self.dim = self.model.visual.output_dim
        
        # Residual Dual-Encoder heads
        self.dropout = nn.Dropout(p=0.3)
        self.img_gate = nn.Linear(self.dim, self.dim)
        self.txt_gate = nn.Linear(self.dim, self.dim)
        
        # Initialize to zero to start at CLIP Zero-Shot baseline
        nn.init.zeros_(self.img_gate.weight)
        nn.init.zeros_(self.img_gate.bias)
        nn.init.zeros_(self.txt_gate.weight)
        nn.init.zeros_(self.txt_gate.bias)
        
    def forward(self, image, text_candidates):
        B = image.shape[0]
        K = len(text_candidates)
        
        flat_texts = [text for sublist in text_candidates for text in sublist]
        text_tokens = clip.tokenize(flat_texts).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image).float()
            text_features = self.model.encode_text(text_tokens).float()

        # Apply residual connection with dropout to mitigate overfitting
        image_features = image_features + self.img_gate(self.dropout(image_features))
        text_features = text_features + self.txt_gate(self.dropout(text_features))

        # L2 Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        C = image_features.shape[-1]
        text_features = text_features.view(K, B, C).transpose(0, 1)
        
        logit_scale = self.model.logit_scale.exp()
        
        # Compute baseline cosine similarity
        # image_features: (B, 1, C), text_features.transpose(1, 2): (B, C, K)
        logits = logit_scale * torch.bmm(
            image_features.unsqueeze(1), 
            text_features.transpose(1, 2)
        ).squeeze(1) # Shape: (B, K)
            
        return logits
