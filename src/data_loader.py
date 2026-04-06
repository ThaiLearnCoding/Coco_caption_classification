import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

class CocoCaptionDataset(Dataset):
    def __init__(self, data_list, image_dir, preprocess, n_captions=5):
        self.data_list = data_list
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.n_captions = n_captions

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_id = item['image_id']
        image_path = os.path.join(self.image_dir, item['file_name'])
        
        image = self.preprocess(Image.open(image_path).convert("RGB"))
        
        # Get ground truth
        gt_caption = item['caption'] # Assuming 'caption' holds the GT
        
        # Get distractors
        distractors = item['distractors']
        
        # Sample distractors to match n_captions - 1
        num_distractors_needed = self.n_captions - 1
        if len(distractors) >= num_distractors_needed:
            selected_distractors = random.sample(distractors, num_distractors_needed)
        else:
            # Handle edge case if not enough distractors (shouldn't happen with proper dataset)
            selected_distractors = distractors + [distractors[0]] * (num_distractors_needed - len(distractors))
            
        captions = [gt_caption] + selected_distractors
        
        # Shuffle captions
        shuffled_indices = list(range(self.n_captions))
        random.shuffle(shuffled_indices)
        
        shuffled_captions = [captions[i] for i in shuffled_indices]
        gt_index = shuffled_indices.index(0) # Index of the ground truth in the shuffled list
        
        return image, shuffled_captions, gt_index, image_id, item['category']

def create_few_shot_splits(data, n_max_shots=32, seed=42):
    class_to_items = defaultdict(list)
    for item in data:
        class_to_items[item['category']].append(item)
        
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

def create_dataloaders(train_data, test_data, image_dir, preprocess, batch_size=32, k=32, n_captions=5, num_workers=4):
    # Sample k shots from the max shots train set
    train_k_data = []
    class_to_train_items = defaultdict(list)
    for item in train_data:
         class_to_train_items[item['category']].append(item)
         
    for cls, items in class_to_train_items.items():
        if len(items) >= k:
             train_k_data.extend(random.sample(items, k))
        else:
             train_k_data.extend(items)
             
    train_dataset = CocoCaptionDataset(train_k_data, image_dir, preprocess, n_captions)
    test_dataset = CocoCaptionDataset(test_data, image_dir, preprocess, n_captions)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader
