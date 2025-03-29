import os
import time
import logging
import warnings
import gc
from datetime import datetime
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# --------------------------
# Configuration & Directories
# --------------------------
REAL_IMG_DIR = "C:/ai/real"  # Contains 360k real images
FAKE_IMG_DIR = "C:/ai/fake"  # Contains 290k fake images

SPLIT_DIR = "splits"
os.makedirs(SPLIT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "checknpoints"
PLOTS_DIR = "plots"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------
# Logging Setup
# --------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# Seed Setting
# --------------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------------
# Early Stopping and Metrics
# --------------------------
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.learning_rates = []

    def update(self, train_loss, val_loss, accuracy, precision, recall, f1_score, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.accuracies.append(accuracy)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1_score)
        self.learning_rates.append(lr)

    def plot_metrics(self, save_dir="plots"):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'losses_{timestamp}.png'))
        plt.close()

        # Plot metrics
        plt.figure(figsize=(10, 6))
        plt.plot(self.accuracies, label='Accuracy')
        plt.plot(self.precisions, label='Precision')
        plt.plot(self.recalls, label='Recall')
        plt.plot(self.f1_scores, label='F1 Score')
        plt.title('Model Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'metrics_{timestamp}.png'))
        plt.close()

        # Plot learning rate
        plt.figure(figsize=(10, 6))
        plt.plot(self.learning_rates)
        plt.title('Learning Rate over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.savefig(os.path.join(save_dir, f'learning_rate_{timestamp}.png'))
        plt.close()

# --------------------------
# Dataset Splitting & Custom Dataset
# --------------------------
def create_or_load_csv_splits(real_dir, fake_dir, split_dir, test_size=0.2, random_state=42):
    train_csv = os.path.join(split_dir, "train_split.csv")
    val_csv = os.path.join(split_dir, "val_split.csv")
    if os.path.exists(train_csv) and os.path.exists(val_csv):
        return train_csv, val_csv
    # Otherwise, create splits
    real_imgs = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    fake_imgs = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_paths = real_imgs + fake_imgs
    labels = [1] * len(real_imgs) + [0] * len(fake_imgs)
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=random_state
    )
    df_train = pd.DataFrame({"image": train_imgs, "label": train_labels})
    df_val = pd.DataFrame({"image": val_imgs, "label": val_labels})
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    logger.info(f"Train split saved to {train_csv} with {len(df_train)} samples.")
    logger.info(f"Validation split saved to {val_csv} with {len(df_val)} samples.")
    return train_csv, val_csv

# --------------------------
# CSVDataset Definition
# --------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image']
        label = int(self.data.iloc[idx]['label'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# --------------------------
# Data Transforms
# --------------------------
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# --------------------------
# DataLoader Creation with Compatibility for Checkpoint Ordering
# --------------------------
def create_train_dataloader(train_csv, batch_size, start_batch=0, ordering=None):
    transform = get_train_transform()
    dataset = CSVDataset(train_csv, transform=transform)
    
    if ordering is not None:
        # If an ordering list was provided (from an earlier checkpoint), use it.
        resumed_ordering = ordering[start_batch:]
        sampler = torch.utils.data.SubsetRandomSampler(resumed_ordering)
    else:
        # For old checkpoints without ordering, generate a deterministic ordering.
        torch.manual_seed(42)  # Seed for torch
        random.seed(42)        # Seed for Python's random module
        full_ordering = list(range(len(dataset)))
        random.shuffle(full_ordering)  # Deterministic shuffle
        resumed_ordering = full_ordering[start_batch:]
        sampler = torch.utils.data.SubsetRandomSampler(resumed_ordering)
        ordering = full_ordering  # Save ordering so future checkpoints can store it.

    num_workers = 0 if os.name != 'nt' else 0  # On Windows, use 0 workers.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True
    )
    return dataloader, ordering

def create_val_dataloader(val_csv, batch_size):
    transform = get_val_transform()
    dataset = CSVDataset(val_csv, transform=transform)
    num_workers = 8 if os.name != 'nt' else 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True
    )
    return dataloader

# --------------------------
# Training Function with Optional Ordering for Compatibility
# --------------------------
def train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, total_epochs, scheduler, start_batch=0, log_interval=100, ordering=None):
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    total_batches = len(train_loader)
    batch_times = []  # To store individual batch times

    for batch_idx, (images, labels) in enumerate(train_loader):
        # Skip batches if resuming mid-epoch
        if batch_idx < start_batch:
            continue

        start_batch_time = time.time()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time = time.time() - start_batch_time
        batch_times.append(batch_time)

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
        targets.extend(labels.cpu().numpy())

        # Log every log_interval batches
        if (batch_idx + 1) % log_interval == 0:
            window = batch_times[-log_interval:]
            avg_batch_time = sum(window) / len(window)
            batches_left = total_batches - (batch_idx + 1)
            eta = avg_batch_time * batches_left
            epoch_progress = (batch_idx + 1) / total_batches * 100
            precision_batch, recall_batch, f1_batch, _ = precision_recall_fscore_support(
                labels.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0
            )
            accuracy_batch = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            logging.info(
                f"Epoch [{epoch+1}/{total_epochs}] Batch [{batch_idx+1}/{total_batches}] "
                f"({epoch_progress:.2f}%) Loss: {loss.item():.6f} | "
                f"Acc: {accuracy_batch:.4f} | Prec: {precision_batch:.4f} | Rec: {recall_batch:.4f} | F1: {f1_batch:.4f} | "
                f"ETA: {eta:.2f}s"
            )

        # Save checkpoint every 500 batches with ordering if available.
        if (batch_idx + 1) % 500 == 0:
            checkpoint_filename = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}batch{batch_idx+1}.pt")
            checkpoint_info = {'batch_idx': batch_idx + 1}
            if ordering is not None:
                checkpoint_info['ordering'] = ordering
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, checkpoint_info, checkpoint_filename)
            logger.info(f"Checkpoint saved at epoch {epoch+1}, batch {batch_idx+1}")

    # Compute average loss over the processed batches
    processed_batches = total_batches - start_batch
    epoch_loss = running_loss / processed_batches if processed_batches > 0 else running_loss
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(targets, predictions)
    return epoch_loss, accuracy, precision, recall, f1

# --------------------------
# Validation Function
# --------------------------
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(targets, predictions)
    return val_loss, accuracy, precision, recall, f1

# --------------------------
# Checkpointing Functions (Compatible with Both Versions)
# --------------------------
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, other_info, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'scaler_state_dict': scaler.state_dict(),
    }
    checkpoint.update(other_info)
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer, scheduler, scaler, device):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        batch_idx = checkpoint.get('batch_idx', 0)
        ordering = checkpoint.get('ordering', None)
        return epoch, batch_idx, ordering
    return 0, 0, None

# --------------------------
# Main Training Loop
# --------------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model, optimizer, scheduler, and scaler
    model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=2)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # Define checkpoint_path and load checkpoint if exists
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        start_epoch, start_batch, saved_ordering = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, scaler, device
        )
    else:
        start_epoch, start_batch, saved_ordering = 0, 0, None

    logger.info(f"Resuming training from epoch {start_epoch+1} and batch {start_batch+1 if start_batch else 1}")

    # Create CSV splits and corresponding DataLoaders
    train_csv, val_csv = create_or_load_csv_splits(REAL_IMG_DIR, FAKE_IMG_DIR, SPLIT_DIR)
    train_loader, ordering = create_train_dataloader(train_csv, BATCH_SIZE, start_batch=start_batch, ordering=saved_ordering)
    val_loader = create_val_dataloader(val_csv, BATCH_SIZE)

    metrics_tracker = MetricsTracker()
    early_stopping = EarlyStopping(patience=2)

    try:
        for epoch in range(start_epoch, EPOCHS):
            epoch_start_time = time.time()

            # For resumed epoch, use current start_batch; for later epochs, start at 0.
            current_start_batch = start_batch if epoch == start_epoch else 0

            train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
                model, train_loader, nn.CrossEntropyLoss(), optimizer, device, scaler,
                epoch, EPOCHS, scheduler, start_batch=current_start_batch, log_interval=100, ordering=ordering
            )

            val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
                model, val_loader, nn.CrossEntropyLoss(), device
            )

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            metrics_tracker.update(
                train_loss, val_loss, val_acc, train_prec, train_rec, val_f1, current_lr
            )

            logger.info(
                f"\nEpoch {epoch+1}/{EPOCHS} Summary:\n"
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n"
                f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}\n"
                f"Train Prec: {train_prec:.4f}, Val Prec: {val_prec:.4f}\n"
                f"Train Rec: {train_rec:.4f}, Val Rec: {val_rec:.4f}\n"
                f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}\n"
                f"Learning Rate: {current_lr:.8f}\n"
                f"Epoch Time: {time.time() - epoch_start_time:.2f}s\n"
            )

            # Save epoch checkpoint (resetting batch_idx to 0)
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_prec': train_prec,
                'val_prec': val_prec,
                'train_rec': train_rec,
                'val_rec': val_rec,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'batch_idx': 0
            }
            if ordering is not None:
                metrics['ordering'] = ordering
            cp_epoch = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, metrics, cp_epoch)
            save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, metrics, checkpoint_path)

            metrics_tracker.plot_metrics(save_dir=PLOTS_DIR)

            early_stopping(val_loss)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # For subsequent epochs, recreate the train_loader without resuming mid-epoch.
            start_batch = 0
            train_loader, ordering = create_train_dataloader(train_csv, BATCH_SIZE, start_batch, ordering)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
    finally:
        final_model_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()
