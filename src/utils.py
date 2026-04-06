import torch
from sklearn.metrics import accuracy_score, f1_score
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import cv2
import numpy as np
import matplotlib.pyplot as plt
import clip
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM, EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    pass

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

def extract_image_attention(model, preprocess, image_path, text_query, device, model_type="vit"):
    """
    Extract attention map using EigenCAM for ViT or GradCAM for ResNet.
    For simplicity, we visualize what the image encoder focuses on.
    """
    try:
        # Load and preprocess image
        original_image = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(original_image).unsqueeze(0).to(device)
        text_tokens = clip.tokenize([text_query]).to(device)
        
        # Prepare image for visualization
        rgb_img = np.float32(original_image) / 255
        rgb_img = cv2.resize(rgb_img, (224, 224))
        
        # Determine target layer based on model type
        if model_type == "vit":
            # For ViT, we typically use the last normalization layer before the final projection
            target_layers = [model.visual.transformer.resblocks[-1].ln_1]
            cam = EigenCAM(model=model.visual, target_layers=target_layers)
        else:
            # For ResNet (RN50), we use the last bottleneck layer
            target_layers = [model.visual.layer4[-1]]
            cam = GradCAM(model=model.visual, target_layers=target_layers)
            
        # Generate CAM
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        return original_image, cam_image
    except Exception as e:
        print(f"CAM extraction failed: {e}")
        return None, None

def plot_prediction_visualizations(subset_data, vit_zs_model, rn50_zs_model, preprocess_vit, preprocess_rn50, device):
    """
    Plot and visualize attention maps using GradCAM / EigenCAM.
    """
    print("\n--- Visualizing Predictions ---")
    if subset_data and len(subset_data) > 0:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("Prediction Visualizations: Original Image vs Attention Map\nTop: ViT (EigenCAM) | Bottom: RN50 (GradCAM)", fontsize=16)
        
        # Select a sample image from subset_data
        sample = subset_data[0]
        img_path = sample['image']
        true_label = sample['binary_label']
        text_query = "an image showing the object" if true_label == 1 else "an image without the object"
        
        # Helper to plot
        def plot_cam(ax_orig, ax_cam, orig_img, cam_img, title):
            if orig_img is not None and cam_img is not None:
                ax_orig.imshow(orig_img)
                ax_orig.set_title("Original")
                ax_orig.axis('off')
                ax_cam.imshow(cam_img)
                ax_cam.set_title(title)
                ax_cam.axis('off')
            else:
                ax_orig.axis('off')
                ax_cam.axis('off')

        # ViT EigenCAM
        vit_orig, vit_cam = extract_image_attention(vit_zs_model, preprocess_vit, img_path, text_query, device, model_type="vit")
        plot_cam(axes[0, 0], axes[0, 1], vit_orig, vit_cam, "ViT EigenCAM")
        
        # ResNet50 GradCAM
        rn50_orig, rn50_cam = extract_image_attention(rn50_zs_model, preprocess_rn50, img_path, text_query, device, model_type="rn50")
        plot_cam(axes[1, 0], axes[1, 1], rn50_orig, rn50_cam, "RN50 GradCAM")
        
        # Visualize another sample
        if len(subset_data) > 1:
            sample2 = subset_data[-1]
            img_path2 = sample2['image']
            true_label2 = sample2['binary_label']
            text_query2 = "an image showing the object" if true_label2 == 1 else "an image without the object"
            
            vit_orig2, vit_cam2 = extract_image_attention(vit_zs_model, preprocess_vit, img_path2, text_query2, device, model_type="vit")
            plot_cam(axes[0, 2], axes[0, 3], vit_orig2, vit_cam2, "ViT EigenCAM (Sample 2)")
            
            rn50_orig2, rn50_cam2 = extract_image_attention(rn50_zs_model, preprocess_rn50, img_path2, text_query2, device, model_type="rn50")
            plot_cam(axes[1, 2], axes[1, 3], rn50_orig2, rn50_cam2, "RN50 GradCAM (Sample 2)")
            
        plt.tight_layout()
        plt.show()
    else:
        print("Data not available for visualization.")
