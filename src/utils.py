import torch
from sklearn.metrics import accuracy_score, f1_score
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# CLIP normalization for ViT and ResNet
# Reference: https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
clip_normalize = Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)

def get_image_transform(image_size=224):
    """
    Standard image transformation pipeline for models like ViT or ResNet.
    """
    return Compose([
        Resize(image_size, interpolation=3), # InterpolationMode.BICUBIC
        CenterCrop(image_size),
        ToTensor(),
        clip_normalize
    ])

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, text_candidates, targets, _, _ = batch
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images, text_candidates)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    return acc, f1

def calculate_model_size_params(model):
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print('Model size: {:.3f}MB'.format(size_all_mb))
    
    return total_params, trainable_params, size_all_mb
