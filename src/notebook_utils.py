import copy
import os
import random
import time

import pandas as pd
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from . import utils


def plot_eda(subset_data, image_dir="../coco_subset_images/images", sample_size=9):
    if not subset_data:
        print("No data available for EDA.")
        return None, None

    counts = {}
    for item in subset_data:
        label = item.get("true_label")
        counts[label] = counts.get(label, 0) + 1

    df = pd.DataFrame(counts.items(), columns=["Category", "Count"]).sort_values(
        "Count", ascending=False
    )

    fig = px.bar(
        df,
        x="Category",
        y="Count",
        title="Data Distribution per Category (COCO Subset)",
        color="Category",
        template="plotly_white",
    )
    fig.show()

    sample_size = min(sample_size, len(subset_data))
    samples = random.sample(subset_data, sample_size)

    rows, cols = 3, 3
    fig_grid, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= len(samples):
            ax.axis("off")
            continue

        sample = samples[idx]
        image_id = sample.get("image_id")
        img_path = os.path.join(image_dir, f"{image_id}.jpg")

        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
            ax.axis("off")

            gt_index = sample.get("ground_truth_idx")
            choices = sample.get("choices", [])
            gt = choices[gt_index] if gt_index is not None else ""

            distractors = [choice for j, choice in enumerate(choices) if j != gt_index]
            distractor = distractors[0] if distractors else "No distractor"

            caption = f"GT: {gt[:30]}...\nDist: {distractor[:30]}..."
            ax.set_title(caption, fontsize=9)
        except Exception:
            ax.set_title("Image missing", color="crimson", fontsize=10)
            ax.axis("off")

    fig_grid.suptitle("Random 3x3 Image Grid with Captions", fontsize=14)
    fig_grid.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return fig, fig_grid


def train_linear_probe(
    model,
    model_name,
    preprocess,
    k,
    train_max_data,
    eval_data,
    config,
    device,
    data_loader,
    num_epochs=10,
    train_mode=True,
    image_dir=None,
    models_dir="../models",
    wandb=None,
    prev_k=None,
    seed=42,
    only_new=False,
    warm_start_from_prev=False,
    warm_start_lr_scale=1.0,
):
    if image_dir is None:
        image_dir = "../coco_subset_images/images"

    training_cfg = config.get("training", {}) if config else {}
    k_train_loader, k_test_loader = data_loader.create_dataloaders(
        train_max_data,
        eval_data,
        image_dir,
        preprocess,
        batch_size=training_cfg.get("batch_size", 32),
        k=k,
        prev_k=prev_k,
        seed=seed,
        only_new=only_new,
    )

    hidden_dim = model.dim // 16
    model.img_gate = nn.Sequential(
        nn.Linear(model.dim, hidden_dim, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, model.dim, bias=False),
    ).to(device)

    model.txt_gate = nn.Sequential(
        nn.Linear(model.dim, hidden_dim, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, model.dim, bias=False),
    ).to(device)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    save_path = os.path.join(models_dir, f"best_{model_name}_{k}shot.pth")

    best_acc = 0.0
    start_epoch = 0

    warm_started = False
    if os.path.exists(save_path):
        print(f"Loading existing checkpoint for {model_name} from {save_path}...")
        checkpoint = torch.load(save_path, map_location=device)
        if "img_gate" in checkpoint:
            model.img_gate.load_state_dict(checkpoint["img_gate"])
            model.txt_gate.load_state_dict(checkpoint["txt_gate"])
            best_acc = checkpoint.get("best_acc", 0.0)
            start_epoch = checkpoint.get("epoch", -1) + 1
        else:
            print("Checkpoint incompatible (legacy format), using initial zeros")
            nn.init.zeros_(model.img_gate[2].weight)
            nn.init.zeros_(model.txt_gate[2].weight)
    elif warm_start_from_prev and prev_k is not None:
        prev_path = os.path.join(models_dir, f"best_{model_name}_{prev_k}shot.pth")
        if os.path.exists(prev_path):
            print(f"Warm-starting {model_name} from {prev_path}...")
            checkpoint = torch.load(prev_path, map_location=device)
            if "img_gate" in checkpoint:
                model.img_gate.load_state_dict(checkpoint["img_gate"])
                model.txt_gate.load_state_dict(checkpoint["txt_gate"])
                warm_started = True
            else:
                print("Checkpoint incompatible (legacy format), using initial zeros")
                nn.init.zeros_(model.img_gate[2].weight)
                nn.init.zeros_(model.txt_gate[2].weight)
    else:
        if not train_mode:
            print(
                f"Warning: No saved checkpoint found at {save_path}. Random Zero-shot inferences will be served."
            )
        else:
            print(f"No checkpoint found. Initializing new Residual Gates for {model_name}.")
        nn.init.zeros_(model.img_gate[2].weight)
        nn.init.zeros_(model.txt_gate[2].weight)

    if not train_mode:
        print(f"--- Inferencing Only for {model_name} k={k} ---")
        eval_acc, f1_macro, f1_micro, f1_weighted = utils.evaluate_model(
            model, k_test_loader, device
        )
        print(
            "Loaded Checkpoint - Eval Acc: {:.4f} - F1 Macro: {:.4f} - F1 Micro: {:.4f} - F1 Weighted: {:.4f}".format(
                eval_acc, f1_macro, f1_micro, f1_weighted
            )
        )
        return eval_acc, f1_macro, f1_micro, f1_weighted

    base_lr = training_cfg.get("learning_rate", 1e-3)
    if warm_started:
        base_lr = base_lr * warm_start_lr_scale

    optimizer = optim.Adam(
        list(model.img_gate.parameters()) + list(model.txt_gate.parameters()),
        lr=base_lr,
        weight_decay=training_cfg.get("weight_decay", 1e-3),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    best_model_state = None

    patience = 5
    epochs_no_improve = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(k_train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images, text_candidates, targets, _, _ = batch
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images, text_candidates)
            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix({"loss": loss.item()})

        scheduler.step()

        train_acc = correct / max(total, 1)
        eval_acc, f1_macro, f1_micro, f1_weighted = utils.evaluate_model(
            model, k_test_loader, device
        )

        print(
            "Epoch {} - Loss: {:.4f} - Train Acc: {:.4f} - Eval Acc: {:.4f} - F1 Macro: {:.4f} - F1 Micro: {:.4f} - F1 Weighted: {:.4f}".format(
                epoch + 1,
                total_loss / len(k_train_loader),
                train_acc,
                eval_acc,
                f1_macro,
                f1_micro,
                f1_weighted,
            )
        )

        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    f"{model_name}_{k}shot_train_loss": total_loss / len(k_train_loader),
                    f"{model_name}_{k}shot_train_acc": train_acc,
                    f"{model_name}_{k}shot_eval_acc": eval_acc,
                    f"{model_name}_{k}shot_eval_f1_macro": f1_macro,
                    f"{model_name}_{k}shot_eval_f1_micro": f1_micro,
                    f"{model_name}_{k}shot_eval_f1_weighted": f1_weighted,
                    f"{model_name}_{k}shot_lr": scheduler.get_last_lr()[0],
                }
            )

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_model_state = {
                "epoch": epoch,
                "best_acc": best_acc,
                "img_gate": copy.deepcopy(model.img_gate.state_dict()),
                "txt_gate": copy.deepcopy(model.txt_gate.state_dict()),
            }
            torch.save(best_model_state, save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs."
            )
            break

    if best_model_state:
        model.img_gate.load_state_dict(best_model_state["img_gate"])
        model.txt_gate.load_state_dict(best_model_state["txt_gate"])

    if wandb is not None and wandb.run is not None:
        artifact = wandb.Artifact(f"{model_name}_{k}shot_model", type="model")
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)

    return best_acc, f1_macro, f1_micro, f1_weighted


def run_few_shot_training(
    vit_lp_model,
    rn50_lp_model,
    preprocess_vit,
    preprocess_rn50,
    train_max_data,
    eval_data,
    config,
    device,
    data_loader,
    k_shots_list=None,
    train_mode=True,
    image_dir=None,
    models_dir="../models",
    wandb=None,
    warm_start=False,
    only_new=False,
    seed=42,
    warm_start_lr_scale=1.0,
):
    if k_shots_list is None:
        k_shots_list = config.get("training", {}).get("k_shots", [8, 16, 32])

    if train_mode and wandb is not None:
        try:
            wandb.login()
            project_name = (
                config.get("wandb", {}).get("project")
                if config and "wandb" in config
                else "coco_multimodal"
            )
            wandb.init(project=project_name, entity=config.get("wandb", {}).get("entity"))
        except Exception as exc:
            print(f"\n[Warning] Weights & Biases failed to initialize: {exc}")

    k_shots_list = sorted(k_shots_list)
    prev_k = None
    for k in k_shots_list:
        print(f"\n--- Model Processor for ViT-B/32 k={k} ---")
        train_linear_probe(
            vit_lp_model,
            "vit_b32",
            preprocess_vit,
            k,
            train_max_data,
            eval_data,
            config,
            device,
            data_loader,
            num_epochs=config.get("training", {}).get("epochs", 25),
            train_mode=train_mode,
            image_dir=image_dir,
            models_dir=models_dir,
            wandb=wandb,
            prev_k=prev_k,
            seed=seed,
            only_new=only_new,
            warm_start_from_prev=warm_start,
            warm_start_lr_scale=warm_start_lr_scale,
        )

        print(f"\n--- Model Processor for RN50 k={k} ---")
        train_linear_probe(
            rn50_lp_model,
            "rn50",
            preprocess_rn50,
            k,
            train_max_data,
            eval_data,
            config,
            device,
            data_loader,
            num_epochs=config.get("training", {}).get("epochs", 25),
            train_mode=train_mode,
            image_dir=image_dir,
            models_dir=models_dir,
            wandb=wandb,
            prev_k=prev_k,
            seed=seed,
            only_new=only_new,
            warm_start_from_prev=warm_start,
            warm_start_lr_scale=warm_start_lr_scale,
        )

        prev_k = k

    if train_mode and wandb is not None and wandb.run is not None:
        wandb.finish()


def evaluate_zero_shot(
    model,
    preprocess,
    train_max_data,
    eval_data,
    config,
    device,
    data_loader,
    image_dir=None,
):
    if not eval_data:
        return 0, 0, 0

    if image_dir is None:
        image_dir = "../coco_subset_images/images"

    _, test_loader = data_loader.create_dataloaders(
        train_max_data,
        eval_data,
        image_dir,
        preprocess,
        batch_size=config.get("training", {}).get("batch_size", 32),
        k=8,
    )

    start_time = time.time()
    acc, f1_macro, f1_micro, f1_weighted = utils.evaluate_model(
        model, test_loader, device
    )
    inference_time = time.time() - start_time

    eval_count = len(eval_data)
    print(f"Zero-Shot Accuracy: {acc:.4f}")
    print(f"Zero-Shot F1 Macro: {f1_macro:.4f}")
    print(f"Zero-Shot F1 Micro: {f1_micro:.4f}")
    print(f"Zero-Shot F1 Weighted: {f1_weighted:.4f}")
    print(f"Inference Time for {eval_count} samples: {inference_time:.4f}s")

    return acc, f1_macro, f1_micro, f1_weighted, inference_time


def _ensure_probe_gates(model, device):
    if not hasattr(model, "img_gate") or not hasattr(model, "txt_gate"):
        hidden_dim = model.dim // 16
        model.img_gate = nn.Sequential(
            nn.Linear(model.dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, model.dim, bias=False),
        ).to(device)

        model.txt_gate = nn.Sequential(
            nn.Linear(model.dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, model.dim, bias=False),
        ).to(device)

        nn.init.zeros_(model.img_gate[2].weight)
        nn.init.zeros_(model.txt_gate[2].weight)
    else:
        model.img_gate = model.img_gate.to(device)
        model.txt_gate = model.txt_gate.to(device)


def _load_probe_checkpoint(model, model_name, k, device, models_dir):
    save_path = os.path.join(models_dir, f"best_{model_name}_{k}shot.pth")
    if not os.path.exists(save_path):
        print(f"Checkpoint not found: {save_path}")
        return False

    checkpoint = torch.load(save_path, map_location=device)
    if "img_gate" not in checkpoint or "txt_gate" not in checkpoint:
        print(f"Incompatible checkpoint format: {save_path}")
        return False

    _ensure_probe_gates(model, device)
    model.img_gate.load_state_dict(checkpoint["img_gate"])
    model.txt_gate.load_state_dict(checkpoint["txt_gate"])
    return True


def evaluate_few_shot(
    model,
    model_name,
    preprocess,
    k,
    train_max_data,
    eval_data,
    config,
    device,
    data_loader,
    image_dir=None,
    models_dir="../models",
):
    if image_dir is None:
        image_dir = "../coco_subset_images/images"

    _, test_loader = data_loader.create_dataloaders(
        train_max_data,
        eval_data,
        image_dir,
        preprocess,
        batch_size=config.get("training", {}).get("batch_size", 32),
        k=k,
    )

    if not _load_probe_checkpoint(model, model_name, k, device, models_dir):
        return None

    start_time = time.time()
    acc, f1_macro, f1_micro, f1_weighted = utils.evaluate_model(
        model, test_loader, device
    )
    inference_time = time.time() - start_time

    eval_count = len(eval_data)
    print(f"{model_name.upper()} {k}-Shot Accuracy: {acc:.4f}")
    print(f"{model_name.upper()} {k}-Shot F1 Macro: {f1_macro:.4f}")
    print(f"{model_name.upper()} {k}-Shot F1 Micro: {f1_micro:.4f}")
    print(f"{model_name.upper()} {k}-Shot F1 Weighted: {f1_weighted:.4f}")
    print(f"Inference Time for {eval_count} samples: {inference_time:.4f}s")

    return acc, f1_macro, f1_micro, f1_weighted, inference_time


def run_evaluation(
    vit_zs_model,
    rn50_zs_model,
    vit_lp_model,
    rn50_lp_model,
    preprocess_vit,
    preprocess_rn50,
    train_max_data,
    eval_data,
    config,
    device,
    data_loader,
    k_shots_list=None,
    image_dir=None,
    models_dir="../models",
):
    if k_shots_list is None:
        k_shots_list = config.get("training", {}).get("k_shots", [8, 16, 32])

    results = []

    print("\n--- ViT-B/32 Zero-Shot Evaluation ---")
    vit_zs_acc, vit_zs_f1_macro, vit_zs_f1_micro, vit_zs_f1_weighted, vit_zs_time = evaluate_zero_shot(
        vit_zs_model,
        preprocess_vit,
        train_max_data,
        eval_data,
        config,
        device,
        data_loader,
        image_dir=image_dir,
    )
    results.append(
        {
            "Model": "ViT-B/32",
            "Shots": 0,
            "Mode": "Zero-shot",
            "Accuracy": vit_zs_acc,
            "F1Macro": vit_zs_f1_macro,
            "F1Micro": vit_zs_f1_micro,
            "F1Weighted": vit_zs_f1_weighted,
            "TimeSec": vit_zs_time,
        }
    )

    print("\n--- RN50 Zero-Shot Evaluation ---")
    rn50_zs_acc, rn50_zs_f1_macro, rn50_zs_f1_micro, rn50_zs_f1_weighted, rn50_zs_time = evaluate_zero_shot(
        rn50_zs_model,
        preprocess_rn50,
        train_max_data,
        eval_data,
        config,
        device,
        data_loader,
        image_dir=image_dir,
    )
    results.append(
        {
            "Model": "RN50",
            "Shots": 0,
            "Mode": "Zero-shot",
            "Accuracy": rn50_zs_acc,
            "F1Macro": rn50_zs_f1_macro,
            "F1Micro": rn50_zs_f1_micro,
            "F1Weighted": rn50_zs_f1_weighted,
            "TimeSec": rn50_zs_time,
        }
    )

    for k in k_shots_list:
        print(f"\n--- ViT-B/32 Few-Shot Evaluation (k={k}) ---")
        vit_res = evaluate_few_shot(
            vit_lp_model,
            "vit_b32",
            preprocess_vit,
            k,
            train_max_data,
            eval_data,
            config,
            device,
            data_loader,
            image_dir=image_dir,
            models_dir=models_dir,
        )
        if vit_res:
            acc, f1_macro, f1_micro, f1_weighted, t = vit_res
            results.append(
                {
                    "Model": "ViT-B/32",
                    "Shots": k,
                    "Mode": "Few-shot",
                    "Accuracy": acc,
                    "F1Macro": f1_macro,
                    "F1Micro": f1_micro,
                    "F1Weighted": f1_weighted,
                    "TimeSec": t,
                }
            )

        print(f"\n--- RN50 Few-Shot Evaluation (k={k}) ---")
        rn50_res = evaluate_few_shot(
            rn50_lp_model,
            "rn50",
            preprocess_rn50,
            k,
            train_max_data,
            eval_data,
            config,
            device,
            data_loader,
            image_dir=image_dir,
            models_dir=models_dir,
        )
        if rn50_res:
            acc, f1_macro, f1_micro, f1_weighted, t = rn50_res
            results.append(
                {
                    "Model": "RN50",
                    "Shots": k,
                    "Mode": "Few-shot",
                    "Accuracy": acc,
                    "F1Macro": f1_macro,
                    "F1Micro": f1_micro,
                    "F1Weighted": f1_weighted,
                    "TimeSec": t,
                }
            )

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("No evaluation results to plot.")
        return results_df

    results_df["Shots"] = results_df["Shots"].astype(int)
    eval_count = max(len(eval_data), 1)
    results_df["TimePerSampleMs"] = (results_df["TimeSec"] / eval_count) * 1000.0
    results_df = results_df.sort_values(["Model", "Shots"]).reset_index(drop=True)

    fig_acc = px.line(
        results_df,
        x="Shots",
        y="Accuracy",
        color="Model",
        markers=True,
        hover_data=["Mode"],
        title="Accuracy vs Shots",
        template="plotly_white",
    )
    fig_acc.show()

    fig_f1 = px.line(
        results_df,
        x="Shots",
        y="F1Macro",
        color="Model",
        markers=True,
        hover_data=["Mode"],
        title="Macro F1 vs Shots",
        template="plotly_white",
    )
    fig_f1.show()

    fig_f1_micro = px.line(
        results_df,
        x="Shots",
        y="F1Micro",
        color="Model",
        markers=True,
        hover_data=["Mode"],
        title="Micro F1 vs Shots",
        template="plotly_white",
    )
    fig_f1_micro.show()

    fig_f1_weighted = px.line(
        results_df,
        x="Shots",
        y="F1Weighted",
        color="Model",
        markers=True,
        hover_data=["Mode"],
        title="Weighted F1 vs Shots",
        template="plotly_white",
    )
    fig_f1_weighted.show()

    fig_time = px.bar(
        results_df,
        x="Shots",
        y="TimePerSampleMs",
        color="Model",
        barmode="group",
        hover_data=["Mode", "TimeSec"],
        title="Inference Time per Sample (ms)",
        template="plotly_white",
    )
    fig_time.show()

    return results_df
