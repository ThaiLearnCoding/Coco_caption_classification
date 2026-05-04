import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

class CocoCaptionDataset(Dataset):
    def __init__(self, data_list, image_dir, preprocess, n_captions=12):
        self.data_list = data_list
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.n_captions = n_captions

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_id = item['image_id']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        image = self.preprocess(Image.open(image_path).convert("RGB"))
        
        # Get all choices and figure out ground truth & distractors
        choices = item['choices']
        gt_idx_orig = item['ground_truth_idx']
        gt_caption = choices[gt_idx_orig]
        
        distractors = [c for i, c in enumerate(choices) if i != gt_idx_orig]
        
        # Sample distractors to match n_captions - 1
        num_distractors_needed = self.n_captions - 1
        if len(distractors) >= num_distractors_needed:
            selected_distractors = random.sample(distractors, num_distractors_needed)
        else:
            # Handle edge case if not enough distractors
            selected_distractors = distractors + [distractors[0]] * (num_distractors_needed - len(distractors))
            
        captions = [gt_caption] + selected_distractors
        
        # Shuffle captions
        shuffled_indices = list(range(self.n_captions))
        random.shuffle(shuffled_indices)
        
        shuffled_captions = [captions[i] for i in shuffled_indices]
        gt_index = shuffled_indices.index(0) # Index of the ground truth in the shuffled list
        
        return image, shuffled_captions, gt_index, image_id, item['true_label']

def create_few_shot_splits(data, n_max_shots=32, seed=42):
    class_to_items = defaultdict(list)
    for item in data:
        class_to_items[item['true_label']].append(item)
        
    train_32_list = []
    test_list = []
    
    for cls, items in class_to_items.items():
        if len(items) > n_max_shots:
            train_cls, test_cls = train_test_split(items, train_size=n_max_shots, random_state=seed)
            train_32_list.extend(train_cls)
            test_list.extend(test_cls)
        else:
            print(f"Warning: Class {cls} has only {len(items)} samples, which is less than {n_max_shots}.")
            test_list.extend(items)
            
    return train_32_list, test_list

def create_dataloaders(
    train_data,
    test_data,
    image_dir,
    preprocess,
    batch_size=32,
    k=32,
    n_captions=12,
    num_workers=2,
    prev_k=None,
    seed=42,
    only_new=False,
):
    # Sample k shots from the max shots train set
    train_k_data = []
    class_to_train_items = defaultdict(list)
    for item in train_data:
        class_to_train_items[item['true_label']].append(item)

    rng = random.Random(seed)
    for cls, items in class_to_train_items.items():
        items_sorted = sorted(items, key=lambda x: x.get('image_id', 0))

        if prev_k is None or prev_k <= 0:
            if len(items_sorted) >= k:
                train_k_data.extend(rng.sample(items_sorted, k))
            else:
                train_k_data.extend(items_sorted)
            continue

        prev_k_eff = min(prev_k, len(items_sorted))
        prev_samples = rng.sample(items_sorted, prev_k_eff)
        if k <= prev_k_eff:
            new_samples = []
        else:
            remaining = [item for item in items_sorted if item not in prev_samples]
            new_samples = rng.sample(remaining, min(k - prev_k_eff, len(remaining)))

        if only_new:
            train_k_data.extend(new_samples)
        else:
            train_k_data.extend(prev_samples + new_samples)

    train_dataset = CocoCaptionDataset(train_k_data, image_dir, preprocess, n_captions)
    test_dataset = CocoCaptionDataset(test_data, image_dir, preprocess, n_captions)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
