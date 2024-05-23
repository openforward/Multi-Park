import os
import numpy as np
import pandas as pd
import cv2
import mmcv
from PIL import Image
import torch
from torch.optim import AdamW, Adam
from transformers import get_scheduler, CLIPProcessor, CLIPModel
from peft import LoraConfig, TaskType, get_peft_model
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Sampler
from torchvision.transforms import (RandomResizedCrop,
                                    ColorJitter,
                                    Compose,
                                    ToTensor,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    ColorJitter,
                                    RandomRotation,
                                    RandomVerticalFlip,
                                    Normalize)

from tqdm.auto import tqdm

class_prompt = {"1": "a photo of smoking",
          "2": "a photo of a shirtless person",
          "3": "a photo of a mouse",
          "4": "a photo of a cat",
          "5": "a photo of a dog",
        "9":"Cover the lid of the trash can",
        "7":"Uncovered trash cans",
        "6":"People wearing masks",
        "8":"The picture is occluded",
}


num_epochs = 20
batch_size = 4
learning_rate = 3e-6

model_name = "C:/models/clip-vit-base-patch32"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name, do_rescale=True)

def create_transforms():
    # Define transformations for training data
    train_transformations = [
        RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # Randomly crops and resizes the image
        RandomHorizontalFlip(p=0.5),  # 50% chance of flipping the image horizontally
        RandomVerticalFlip(p=0.5),    # 50% chance of flipping the image vertically
        RandomRotation(degrees=15),   # Randomly rotates the image within a 15 degree range
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly changes brightness, contrast, saturation, and hue
        ToTensor()  # Converts the image to a PyTorch Tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes the tensor with ImageNet's mean and std
    ]

    # Define transformations for validation data
    val_transformations = [
        ToTensor()  # Converts the image to a PyTorch Tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes the tensor with ImageNet's mean and std
    ]

    # Create Compose objects for transformations
    transform = Compose(train_transformations)
    val_transform = Compose(val_transformations)

    return transform, val_transform

# Get the transformations
transform, val_transform = create_transforms()

class CustomImageDataset(ImageFolder):
    def __init__(self, root_dir, transform=None):
        super(CustomImageDataset, self).__init__(root=root_dir, transform=transform)
        self.class_to_text = {v: k for k, v in self.class_to_idx.items()}

    def __getitem__(self, idx):
        img, class_idx = super(CustomImageDataset, self).__getitem__(idx)
        class_text = self.class_to_text[class_idx]
        return img, class_prompt[class_text]

# Usage example
# dataset = CustomImageDataset(root='path/to/dataset', transform=your_transforms)


def collate_fn(batch):
    # Unzip the batch into separate lists for images and texts
    images, texts = zip(*batch)

    # Process the batch data
    inputs = processor(text=list(texts), images=list(images), return_tensors="pt", padding=True)

    return inputs


# train model function
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, return_loss=True)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, return_loss=True)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, num_epochs, device):
    best_score = -np.inf
    score_list = []
    progress_bar = tqdm(total=num_epochs * len(train_dataloader), desc="Training")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        val_loss = validate_epoch(model, val_dataloader, device)
        lr_scheduler.step()
        progress_bar.update(len(train_dataloader))

        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        score = -val_loss
        score_list.append(score)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), "./clip_best.pth")

        if len(score_list) >= 5 and all(score_list[i] > score_list[i + 1] for i in range(-5, -1)):
            print(f"Early stop, best score is: {-best_score:.4f}")
            break

    progress_bar.close()


train_dataset = CustomImageDataset(root_dir=os.path.join("E:/数据样例"), transform=transform)
val_dataset = CustomImageDataset(root_dir=os.path.join("./val"), transform=val_transform)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn=collate_fn)


model.to(device)
num_training_steps = num_epochs * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
train_model(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, num_epochs, device)