import gradio as gr
import json
import random
import os
import plotly.graph_objects as go

def load_data(subset_path):
    with open(subset_path, 'r') as f:
        data = json.load(f)
    return data

def get_random_sample(data, n_captions):
    # Select random image
    sample = random.choice(data)
    
    # Extract ground truth and distractors based on the JSON structure from the notebook
    gt_idx = sample.get('ground_truth_idx', 0)
    choices = sample.get('choices', [])
    gt_caption = choices[gt_idx] if choices else "Dummy Caption"
    
    distractors = [c for i, c in enumerate(choices) if i != gt_idx]
    
    # Needs n_captions - 1 distractors
    num_distractors_needed = n_captions - 1
    if len(distractors) >= num_distractors_needed:
        selected_distractors = random.sample(distractors, num_distractors_needed)
    else:
        # Fallback if not enough distractors
        selected_distractors = distractors + ["Dummy distractor"] * (num_distractors_needed - len(distractors))
        
    captions = [gt_caption] + selected_distractors
    
    # Shuffle
    random.shuffle(captions)
    
    # Construct image path using 'image_id'
    img_id = sample.get('image_id', 'dummy')
    image_path = os.path.join('coco_subset_images', f"{img_id}.jpg")
    
    # Generate dummy line plot for models performances vs k-shots
    # This will be replaced by actual data in a full implementation
    fig = go.Figure()
    
    k_shots = [0, 8, 16, 32]
    vit_acc = [50 + n_captions*2, 60, 75, 85]
    rn50_acc = [45 + n_captions*2, 55, 68, 78]
    
    fig.add_trace(go.Scatter(x=k_shots, y=vit_acc, mode='lines+markers', name='ViT-B/32'))
    fig.add_trace(go.Scatter(x=k_shots, y=rn50_acc, mode='lines+markers', name='ResNet50'))
    
    fig.update_layout(title=f'Model Accuracy vs K-Shots (N={n_captions})',
                      xaxis_title='K-Shots',
                      yaxis_title='Accuracy (%)')
    
    return image_path, captions, fig

# Gradio Interface
with gr.Blocks(title="COCO Multimodal Zero-shot Classification") as demo:
    gr.Markdown("# COCO Multimodal Zero-shot / Few-shot Classification")
    gr.Markdown("This app demonstrates evaluating CLIP models (ViT & ResNet) to classify an image from N candidate captions where 1 is the ground-truth and others are distractors.")
    
    with gr.Row():
        with gr.Column():
            n_slider = gr.Slider(minimum=3, maximum=12, step=1, value=12, label="Number of Captions (N)")
            shuffle_btn = gr.Button("Shuffle & Load Random Sample")
            
            image_out = gr.Image(label="Input Image")
            captions_out = gr.Dataframe(headers=["Candidate Captions"], datatype=["str"], label="N Captions")
            
        with gr.Column():
            plot_out = gr.Plot(label="Performance tracking over k-shots")
            
            gr.Markdown("### Interpretability (Examples)")
            gr.Markdown("**Attention Rollout & GradCam** visualizations will be displayed here for real inferences.")

    # Load data for quick demo
    try:
        data = load_data('coco_subset_images/coco_multimodal_subset.json')
    except:
        # Fallback to dummy data using correct keys
        data = [{
            "image_id": "dummy", 
            "choices": ["Dummy Caption"] + [f"Dummy distractor {i}" for i in range(1, 6)],
            "ground_truth_idx": 0
        }]
        
    def update_view(n_captions):
        if len(data) > 0:
            img_path, caps, fig = get_random_sample(data, n_captions)
            return img_path, [[c] for c in caps], fig
        return None, [], None

    shuffle_btn.click(fn=update_view, inputs=[n_slider], outputs=[image_out, captions_out, plot_out])

if __name__ == "__main__":
    demo.launch(share=False)
