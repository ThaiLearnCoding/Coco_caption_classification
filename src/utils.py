import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import cv2
import numpy as np
import matplotlib.pyplot as plt
import clip
from clip.simple_tokenizer import SimpleTokenizer
from PIL import Image

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
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    f1_micro = f1_score(all_targets, all_preds, average='micro')
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')

    return acc, f1_macro, f1_micro, f1_weighted

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
        from pytorch_grad_cam import GradCAM, EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("Please restart runtime or install grad-cam: !pip install grad-cam opencv-python-headless")
        return None, None

    try:
        # If wrapped in our custom classifier, get the base model
        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model
            
        # Temporarily cast to float32 to avoid GradCAM HalfTensor errors
        prev_dtype = next(base_model.parameters()).dtype
        base_model.float()

        # Load and preprocess image
        original_image = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(original_image).unsqueeze(0).to(device).float()
        text_tokens = clip.tokenize([text_query]).to(device)
        
        # Prepare image for visualization
        rgb_img = np.float32(original_image) / 255
        rgb_img = cv2.resize(rgb_img, (224, 224))
        
        # Determine target layer based on model type
        def reshape_transform(tensor, height=7, width=7):
            # For CLIP ViT: output shape is [50, 1, 768] (Seq_len, Batch, Channels)
            tensor = tensor.permute(1, 0, 2) # [1, 50, 768]
            result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2)) 
            result = result.transpose(2, 3).transpose(1, 2)
            return result

        if model_type == "vit":
            # For ViT, we use the last normalization layer before the final projection
            target_layers = [base_model.visual.transformer.resblocks[-1].ln_1]
            cam = EigenCAM(model=base_model.visual, target_layers=target_layers, reshape_transform=reshape_transform)
        else:
            # For ResNet (RN50), we use the last bottleneck layer
            target_layers = [base_model.visual.layer4[-1]]
            cam = GradCAM(model=base_model.visual, target_layers=target_layers)
            
        # Generate CAM
        grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, (rgb_img.shape[1], rgb_img.shape[0]))
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Revert model dtype
        base_model.to(prev_dtype)

        return original_image, cam_image
    except Exception as e:
        print(f"CAM extraction failed: {e}")
        return None, None


def extract_text_attention_rollout(model, text_query, device, max_tokens=32):
    """
    Extract attention rollout from the text transformer to get token-level importance.
    """
    try:
        if hasattr(model, 'model'):
            base_model = model.model
        else:
            base_model = model

        base_model.eval()
        try:
            tokenizer = SimpleTokenizer()
        except Exception:
            tokenizer = None

        text_tokens = clip.tokenize([text_query]).to(device)
        token_ids = text_tokens[0].detach().cpu().tolist()

        # Locate end-of-text token if present
        eot_token = 49407
        if eot_token in token_ids:
            seq_len = token_ids.index(eot_token) + 1
        else:
            seq_len = len(token_ids)

        attn_mats = []

        def _attn_hook(module, inputs, output):
            x = inputs[0]
            x = x.float()
            attn_mask = None
            if len(inputs) > 1:
                attn_mask = inputs[1]
            elif hasattr(module, 'attn_mask'):
                attn_mask = module.attn_mask

            if attn_mask is not None:
                attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
                if attn_mask.shape[0] != x.shape[0]:
                    attn_mask = attn_mask[:x.shape[0], :x.shape[0]]

            in_proj_weight = module.in_proj_weight
            in_proj_bias = module.in_proj_bias
            out_proj_weight = module.out_proj.weight
            out_proj_bias = module.out_proj.bias
            if in_proj_weight.dtype != x.dtype:
                in_proj_weight = in_proj_weight.to(dtype=x.dtype)
            if in_proj_bias is not None and in_proj_bias.dtype != x.dtype:
                in_proj_bias = in_proj_bias.to(dtype=x.dtype)
            if out_proj_weight.dtype != x.dtype:
                out_proj_weight = out_proj_weight.to(dtype=x.dtype)
            if out_proj_bias is not None and out_proj_bias.dtype != x.dtype:
                out_proj_bias = out_proj_bias.to(dtype=x.dtype)

            embed_dim = module.in_proj_weight.shape[1]
            try:
                _, attn_weights = F.multi_head_attention_forward(
                    query=x,
                    key=x,
                    value=x,
                    embed_dim_to_check=embed_dim,
                    num_heads=module.num_heads,
                    in_proj_weight=in_proj_weight,
                    in_proj_bias=in_proj_bias,
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0.0,
                    out_proj_weight=out_proj_weight,
                    out_proj_bias=out_proj_bias,
                    training=module.training,
                    key_padding_mask=None,
                    need_weights=True,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=False,
                    average_attn_weights=False,
                )
            except TypeError:
                _, attn_weights = F.multi_head_attention_forward(
                    query=x,
                    key=x,
                    value=x,
                    embed_dim_to_check=embed_dim,
                    num_heads=module.num_heads,
                    in_proj_weight=in_proj_weight,
                    in_proj_bias=in_proj_bias,
                    bias_k=None,
                    bias_v=None,
                    add_zero_attn=False,
                    dropout_p=0.0,
                    out_proj_weight=out_proj_weight,
                    out_proj_bias=out_proj_bias,
                    training=module.training,
                    key_padding_mask=None,
                    need_weights=True,
                    attn_mask=attn_mask,
                    use_separate_proj_weight=False,
                )

            if attn_weights.dim() == 2:
                attn_weights = attn_weights.unsqueeze(0).unsqueeze(0)
            elif attn_weights.dim() == 3:
                attn_weights = attn_weights.unsqueeze(1)
            attn_mats.append(attn_weights.detach())

        hooks = []
        for block in base_model.transformer.resblocks:
            hooks.append(block.attn.register_forward_hook(_attn_hook))

        with torch.no_grad():
            _ = base_model.encode_text(text_tokens)

        for hook in hooks:
            hook.remove()

        if not attn_mats:
            return [], []

        seq_len = min(seq_len, attn_mats[0].shape[-1])
        rollout = torch.eye(seq_len, device=device)
        for attn in attn_mats:
            attn = attn[:, :, :seq_len, :seq_len]
            attn_mean = attn.mean(dim=1)[0]
            attn_mean = attn_mean / (attn_mean.sum(dim=-1, keepdim=True) + 1e-8)
            attn_mean = (attn_mean + torch.eye(seq_len, device=device)) / 2.0
            rollout = attn_mean @ rollout

        scores = rollout[0].detach().cpu().tolist()

        tokens = []
        token_scores = []
        for idx, token_id in enumerate(token_ids[:seq_len]):
            if token_id in (0, 49406, 49407):
                continue
            if tokenizer is not None:
                token_text = tokenizer.decode([token_id]).replace("</w>", "").strip()
            else:
                token_text = f"tok_{token_id}"
            if token_text:
                tokens.append(token_text)
                token_scores.append(scores[idx])

        if max_tokens and len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            token_scores = token_scores[:max_tokens]

        return tokens, token_scores
    except Exception as e:
        print(f"Text attention rollout failed: {e}")
        return [], []


def plot_text_attention(ax, tokens, scores, title):
    """
    Plot token-level attention as a colored bar chart.
    """
    if not tokens:
        ax.set_title(f"{title}\n(no tokens)")
        ax.axis('off')
        return

    scores = np.array(scores, dtype=float)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    colors = plt.cm.viridis(scores)
    ax.bar(range(len(tokens)), scores, color=colors)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=10)

import os

def plot_prediction_visualizations(subset_data, vit_zs_model, rn50_zs_model, preprocess_vit, preprocess_rn50, device, image_dir='../coco_subset_images/images'):
    """
    Plot and visualize attention maps using GradCAM / EigenCAM.
    """
    print("\n--- Visualizing Predictions ---")
    if subset_data and len(subset_data) > 0:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("Prediction Visualizations: Original Image vs Attention Map\nTop: ViT (EigenCAM) | Bottom: RN50 (GradCAM)", fontsize=16)
        
        # Select a sample image from subset_data
        sample = subset_data[0]
        img_path = os.path.join(image_dir, f"{sample['image_id']}.jpg")
        text_query = sample['choices'][sample['ground_truth_idx']]
        
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
            img_path2 = os.path.join(image_dir, f"{sample2['image_id']}.jpg")
            text_query2 = sample2['choices'][sample2['ground_truth_idx']]
            
            vit_orig2, vit_cam2 = extract_image_attention(vit_zs_model, preprocess_vit, img_path2, text_query2, device, model_type="vit")
            plot_cam(axes[0, 2], axes[0, 3], vit_orig2, vit_cam2, "ViT EigenCAM (Sample 2)")
            
            rn50_orig2, rn50_cam2 = extract_image_attention(rn50_zs_model, preprocess_rn50, img_path2, text_query2, device, model_type="rn50")
            plot_cam(axes[1, 2], axes[1, 3], rn50_orig2, rn50_cam2, "RN50 GradCAM (Sample 2)")
            
        plt.tight_layout()
        plt.show()

        # Text attention rollout visualization
        fig_txt, axes_txt = plt.subplots(2, 2, figsize=(16, 8))
        fig_txt.suptitle("Text Attention Rollout (Token Importance)", fontsize=14)

        vit_tokens, vit_scores = extract_text_attention_rollout(vit_zs_model, text_query, device)
        plot_text_attention(axes_txt[0, 0], vit_tokens, vit_scores, "ViT Text Attention (Sample 1)")

        rn_tokens, rn_scores = extract_text_attention_rollout(rn50_zs_model, text_query, device)
        plot_text_attention(axes_txt[1, 0], rn_tokens, rn_scores, "RN50 Text Attention (Sample 1)")

        if len(subset_data) > 1:
            vit_tokens2, vit_scores2 = extract_text_attention_rollout(vit_zs_model, text_query2, device)
            plot_text_attention(axes_txt[0, 1], vit_tokens2, vit_scores2, "ViT Text Attention (Sample 2)")

            rn_tokens2, rn_scores2 = extract_text_attention_rollout(rn50_zs_model, text_query2, device)
            plot_text_attention(axes_txt[1, 1], rn_tokens2, rn_scores2, "RN50 Text Attention (Sample 2)")
        else:
            axes_txt[0, 1].axis('off')
            axes_txt[1, 1].axis('off')

        fig_txt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
    else:
        print("Data not available for visualization.")

def plot_failure_cases(model, dataloader, device, image_dir, num_cases=4):
    """
    Find and plot failure cases from the dataloader.
    """
    model.eval()
    failures = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, text_candidates, targets, image_ids, _ = batch
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images, text_candidates)
            preds = torch.argmax(logits, dim=1)
            
            for i in range(len(preds)):
                if preds[i] != targets[i]:
                    img_path = os.path.join(image_dir, f"{image_ids[i]}.jpg")
                    # Dataloader gives text_candidates as a list of tuples, where text_candidates[j] is the j-th candidate for all items in batch
                    # So text_candidates[j][i] is the j-th candidate for the i-th item in the batch.
                    gt_idx = targets[i].item()
                    pred_idx = preds[i].item()
                    candidates = [text_candidates[j][i] for j in range(len(text_candidates))]
                    gt_text = text_candidates[gt_idx][i]
                    pred_text = text_candidates[pred_idx][i]

                    failures.append((img_path, gt_text, pred_text, candidates))
                    
                    if len(failures) >= num_cases:
                        break
            if len(failures) >= num_cases:
                break
                
    if len(failures) == 0:
        print("No failure cases found!")
        return
        
    num_cases = min(num_cases, len(failures))
    fig, axes = plt.subplots(1, num_cases, figsize=(6 * num_cases, 6))
    if num_cases == 1:
        axes = [axes]
        
    import textwrap
    for idx, (img_path, gt_text, pred_text, candidates) in enumerate(failures):
        ax = axes[idx]
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis('off')

        wrapped_gt = "\\n".join(textwrap.wrap(gt_text, width=40))
        wrapped_pred = "\\n".join(textwrap.wrap(pred_text, width=40))

        ax.text(
            0.0,
            1.12,
            f"GT: {wrapped_gt}",
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=10,
            color='green',
            clip_on=False
        )
        ax.text(
            0.0,
            1.02,
            f"Pred: {wrapped_pred}",
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=10,
            color='red',
            clip_on=False
        )

        plot_text_embedding_map(
            candidates,
            gt_text,
            pred_text,
            model,
            device,
            title=f"Text Similarity Map (Case {idx + 1})"
        )

    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    plt.show()


def plot_text_embedding_map(
    texts,
    gt_text,
    pred_text,
    model,
    device,
    title="Text Similarity Map",
    method="pca",
):
    if not texts or len(texts) < 2:
        print("Not enough texts to plot embedding map.")
        return

    base_model = model.model if hasattr(model, 'model') else model
    base_model.eval()

    tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_feats = base_model.encode_text(tokens).float()
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.detach().cpu().numpy()

    if method == "tsne" and len(texts) >= 3:
        reducer = TSNE(n_components=2, init="random", perplexity=min(10, len(texts) - 1), random_state=42)
        coords = reducer.fit_transform(text_feats)
    else:
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(text_feats)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = []
    markers = []
    for text in texts:
        if text == gt_text:
            colors.append('#16a34a')
            markers.append('o')
        elif text == pred_text:
            colors.append('#dc2626')
            markers.append('X')
        else:
            colors.append('#94a3b8')
            markers.append('o')

    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, color=colors[i], marker=markers[i], s=80, alpha=0.9)
        ax.text(x + 0.01, y + 0.01, str(i), fontsize=9)

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Ground Truth', markerfacecolor='#16a34a', markersize=8),
        plt.Line2D([0], [0], marker='X', color='w', label='Predicted', markerfacecolor='#dc2626', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Distractor', markerfacecolor='#94a3b8', markersize=8),
    ]
    ax.legend(handles=legend_handles, loc='best', fontsize=8)

    plt.tight_layout()
    plt.show()

