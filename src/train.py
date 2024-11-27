import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTResNet
import datetime
import os
import glob
from utils import TrainingVisualizer
import random
import numpy as np
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class CurriculumLoss(nn.Module):
    def __init__(self, init_smoothing=0.1):
        super().__init__()
        self.smoothing = init_smoothing
        self.epoch = 0

    def forward(self, pred, target):
        # Progressive label smoothing
        current_smoothing = max(0.01, self.smoothing * (1 - self.epoch/20))
        n_classes = pred.size(1)
        
        # One-hot encoding with smoothing
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(current_smoothing / (n_classes - 1))
        smooth_target.scatter_(1, target.unsqueeze(1), 1 - current_smoothing)
        
        # Compute loss with confidence penalty
        log_prob = F.log_softmax(pred, dim=1)
        loss = -torch.sum(smooth_target * log_prob, dim=1)
        
        # Add confidence penalty for high probability predictions
        probs = torch.exp(log_prob)
        entropy = -torch.sum(probs * log_prob, dim=1)
        confidence_penalty = 0.1 * (1 - entropy/np.log(n_classes))
        
        return (loss + confidence_penalty).mean()

    def update_epoch(self, epoch):
        self.epoch = epoch

def get_batch_size(epoch):
    if epoch < 2:
        return 256
    elif epoch < 5:
        return 128
    elif epoch < 10:
        return 64
    elif epoch < 15:
        return 32
    else:
        return 16

def train():
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Enhanced data augmentation with more focused transformations
    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=20,  # More rotation
            translate=(0.2, 0.2),  # More translation
            scale=(0.8, 1.2),  # More scaling
            shear=15,  # More shear
            fill=0
        ),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.3))
        ], p=0.2),
        transforms.RandomInvert(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomApply([
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.015)  # More noise
        ], p=0.25)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Data loading
    train_dataset = datasets.MNIST(root='./data', train=True, transform=train_transform, download=True)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
    val_dataset.dataset.transform = val_transform
    test_dataset = datasets.MNIST(root='./data', train=False, transform=val_transform)

    # Smaller batch size for better generalization
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=0)

    # Calculate base steps per epoch before creating train_loader
    base_steps_per_epoch = len(train_dataset) // 64  # Using base batch size of 64

    # Initialize model and SWA model
    model = MNISTResNet().to(device)
    swa_model = AveragedModel(model)
    
    # Initialize criterion
    criterion = CurriculumLoss(init_smoothing=0.1)
    
    # Initialize optimizer and schedulers
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.003,
        weight_decay=0.03,
        betas=(0.9, 0.999)
    )

    # Use Cosine Annealing for first 10 epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    # SWA scheduler starts after 10 epochs
    swa_scheduler = SWALR(optimizer, swa_lr=0.001)
    
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    visualizer = TrainingVisualizer()
    best_acc = 0
    best_model_path = None
    swa_start = 10  # Start SWA after 10 epochs

    # Training loop
    for epoch in range(20):
        model.train()
        criterion.update_epoch(epoch)
        
        # Update batch size but keep original train_loader steps
        batch_size = get_batch_size(epoch)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=0)
        
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            if i >= base_steps_per_epoch:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # Modified augmentation mix
            if random.random() < 0.5:
                if random.random() < 0.5:
                    # CutMix with smaller cuts
                    lam = np.random.beta(0.5, 0.5)
                    rand_index = torch.randperm(images.size(0)).to(device)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                    images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                else:
                    # Less aggressive Mixup
                    lam = np.random.beta(0.4, 0.4)
                    rand_index = torch.randperm(images.size(0)).to(device)
                    images = lam * images + (1 - lam) * images[rand_index]
                
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                    loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[rand_index])
            else:
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Gradient accumulation for larger effective batch size
            loss = loss / 4
            scaler.scale(loss).backward()
            
            if (i + 1) % 4 == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Update SWA model if in SWA phase
                if epoch >= swa_start:
                    swa_model.update_parameters(model)

        # Step the appropriate scheduler
        if epoch < swa_start:
            scheduler.step()
        else:
            swa_scheduler.step()
            
        # Calculate epoch metrics only if we have processed some batches
        if train_total > 0:
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
        else:
            train_loss = float('inf')
            train_acc = 0.0

        # Enhanced validation with weighted TTA
        def validate_with_tta(model, images, labels):
            outputs = model(images)
            
            # TTA with minimal augmentations
            augmentations = [
                lambda x: torch.roll(x, shifts=1, dims=2),
                lambda x: torch.roll(x, shifts=-1, dims=2),
                lambda x: x + torch.randn_like(x) * 0.005  # Very small noise
            ]
            
            for aug in augmentations:
                aug_images = aug(images)
                outputs += model(aug_images)
            
            outputs /= (len(augmentations) + 1)
            return outputs

        # Validation loop with enhanced TTA
        if epoch >= swa_start:
            # Update SWA batch norm statistics
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            # Evaluate with SWA model
            swa_model.eval()
            val_model = swa_model
        else:
            model.eval()
            val_model = model

        with torch.no_grad():
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = validate_with_tta(val_model, images, labels)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            visualizer.update(epoch + 1, train_loss, val_loss, train_acc, val_acc, current_lr)
            print(f"{epoch+1:3d}    {train_loss:.4f}    {val_loss:.4f}    {train_acc:6.2f}%    {val_acc:6.2f}%    {current_lr:.6f}")
            
            # Save best model (either regular or SWA)
            if val_acc > best_acc:
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                best_acc = val_acc
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = f'mnist_model_{timestamp}_acc{val_acc:.2f}.pth'
                
                # Save appropriate model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': swa_model.state_dict() if epoch >= swa_start else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': swa_scheduler.state_dict() if epoch >= swa_start else scheduler.state_dict(),
                    'best_acc': best_acc,
                    'is_swa': epoch >= swa_start,
                    'scaler_state_dict': scaler.state_dict(),
                }, best_model_path)

        if val_acc > 99.5:
            print("\nReached target accuracy. Stopping training.")
            break

    print("\nTraining completed. Best validation accuracy: {:.2f}%".format(best_acc))
    visualizer.plot_metrics()
    return best_acc

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

if __name__ == "__main__":
    for model_file in glob.glob('mnist_model_*.pth'):
        os.remove(model_file)
    train() 