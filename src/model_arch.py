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

class LinearProbeClassifier(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu", n_captions=5):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        
        # Freeze CLIP backbone
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        
        # Extract features dim based on model
        self.dim = self.model.visual.output_dim
        
        # Head that takes combined image+text features to similarity score or logits
        # We process each candidate and output a score
        # Specifically, we follow the description: "Train the simple Linear Layer using Cross-Entropy Loss over the similarity scores of the k candidates."
        
        # This implementation applies linear probe over dot product of image and text features
        self.linear_head = nn.Linear(1, 1) 
        
    def forward(self, image, text_candidates):
        B = image.shape[0]
        K = len(text_candidates)
        
        flat_texts = [text for sublist in text_candidates for text in sublist]
        text_tokens = clip.tokenize(flat_texts).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            C = image_features.shape[-1]
            text_features = text_features.view(K, B, C).transpose(0, 1)
            
            # Compute baseline cosine similarity
            # image_features: (B, 1, C), text_features.transpose(1, 2): (B, C, K)
            cosine_sims = torch.bmm(
                image_features.unsqueeze(1), 
                text_features.transpose(1, 2)
            ).squeeze(1) # Shape: (B, K)
            
        # Reshape to apply linear layer independently to each similarity score
        # Then we use CE Loss over the candidates
        cosine_sims_reshaped = cosine_sims.view(-1, 1).to(torch.float32) # (B*K, 1)
        logits_reshaped = self.linear_head(cosine_sims_reshaped) # (B*K, 1)
        logits = logits_reshaped.view(B, K) # (B, K)
            
        return logits
