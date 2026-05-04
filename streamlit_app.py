import os
import tempfile

import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt

from src import model_arch, utils


st.set_page_config(page_title="COCO Caption Classification Demo", layout="wide")


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model(model_name, mode, k_shot, device, models_dir):
    if mode == "Zero-shot":
        model = model_arch.CLIPZeroShotClassifier(model_name=model_name, device=device)
        return model

    model = model_arch.ResidualDualProbeClassifier(model_name=model_name, device=device, n_captions=5)
    ckpt_key = "vit_b32" if model_name == "ViT-B/32" else "rn50"
    ckpt_path = os.path.join(models_dir, f"best_{ckpt_key}_{k_shot}shot.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    if "img_gate" not in checkpoint or "txt_gate" not in checkpoint:
        raise ValueError(f"Checkpoint incompatible: {ckpt_path}")

    model.img_gate.load_state_dict(checkpoint["img_gate"])
    model.txt_gate.load_state_dict(checkpoint["txt_gate"])
    model.eval()
    return model


def predict(model, image, captions, device):
    preprocess = model.preprocess
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text_candidates = [[caption] for caption in captions]

    with torch.no_grad():
        logits = model(image_tensor, text_candidates)
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=-1).item())
        pred_score = float(probs[0, pred_idx].item())

    return pred_idx, pred_score


def render_attention(model, image, text_query, model_type, device):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        orig, cam = utils.extract_image_attention(
            model, model.preprocess, tmp_path, text_query, device, model_type=model_type
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    tokens, scores = utils.extract_text_attention_rollout(model, text_query, device)

    fig, ax = plt.subplots(figsize=(6, 3))
    utils.plot_text_attention(ax, tokens, scores, "Text Attention Rollout")
    fig.tight_layout()

    return orig, cam, fig


st.title("COCO Caption Classification Demo")
st.caption("Zero-shot and few-shot caption matching with image + text attention")

with st.sidebar:
    st.header("Configuration")
    device = st.selectbox("Device", options=[_get_device(), "cpu"], index=0)
    mode = st.selectbox("Mode", options=["Zero-shot", "Few-shot"], index=0)
    model_choices = ["ViT-B/32", "RN50"]
    selected_models = st.multiselect("Models", options=model_choices, default=model_choices)
    k_shot = st.selectbox("Few-shot k", options=[2, 4, 8, 16, 32], index=2)
    models_dir = st.text_input("Checkpoints directory", value="models")

st.subheader("Inputs")
image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

caption_text = st.text_area(
    "Candidate captions (one per line)",
    value="A bear in the forest\nA zebra crossing the road\nAn elephant by the river",
    height=140,
)

captions = [line.strip() for line in caption_text.splitlines() if line.strip()]
if not captions:
    st.warning("Please enter at least 1 caption.")

if captions:
    gt_option = st.selectbox("Ground-truth caption (optional)", ["None"] + captions)
    gt_idx = captions.index(gt_option) if gt_option != "None" else None
else:
    gt_idx = None

run_btn = st.button("Run Inference")

if run_btn:
    if image_file is None:
        st.error("Please upload an image.")
    elif len(captions) < 2:
        st.error("Please provide at least 2 candidate captions.")
    elif not selected_models:
        st.error("Select at least one model.")
    else:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Input Image", use_column_width=True)

        cols = st.columns(len(selected_models))
        for idx, model_name in enumerate(selected_models):
            with cols[idx]:
                st.markdown(f"### {model_name}")
                try:
                    model = load_model(model_name, mode, k_shot, device, models_dir)
                except Exception as exc:
                    st.error(str(exc))
                    continue

                pred_idx, pred_score = predict(model, image, captions, device)
                pred_caption = captions[pred_idx]

                st.markdown("**Predicted caption:**")
                st.markdown(f"`{pred_caption}` (score: {pred_score:.4f})")

                st.markdown("**Captions:**", unsafe_allow_html=True)
                for i, cap in enumerate(captions):
                    color = "#0f172a"
                    if i == pred_idx:
                        if gt_idx is None or i == gt_idx:
                            color = "#16a34a"
                        else:
                            color = "#dc2626"
                    st.markdown(
                        f"<div style='color:{color}; margin-bottom:4px;'>[{i}] {cap}</div>",
                        unsafe_allow_html=True,
                    )

                model_type = "vit" if model_name == "ViT-B/32" else "rn50"
                orig, cam, fig = render_attention(
                    model, image, pred_caption, model_type, device
                )

                if orig is not None and cam is not None:
                    st.markdown("**Image attention:**")
                    st.image([orig, cam], caption=["Original", "Attention"], width=260)

                st.markdown("**Text attention:**")
                st.pyplot(fig)

                if gt_idx is not None and gt_idx != pred_idx:
                    gt_caption = captions[gt_idx]
                    st.markdown("**Ground-truth attention:**")
                    orig_gt, cam_gt, fig_gt = render_attention(
                        model, image, gt_caption, model_type, device
                    )
                    if orig_gt is not None and cam_gt is not None:
                        st.image([orig_gt, cam_gt], caption=["GT Original", "GT Attention"], width=260)
                    st.pyplot(fig_gt)
