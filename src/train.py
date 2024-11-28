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

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Basic transforms - only normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Data loading
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Initialize model, criterion, optimizer
    model = MNISTResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=20,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos'
    )

    # Training visualization
    visualizer = TrainingVisualizer()
    best_acc = 0
    best_model_path = None

    # Training loop
    print("\nEpoch   Train Loss  Val Loss    Train Acc   Val Acc      LR")
    print("--------------------------------------------------------")
    
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch results
        print(f"{epoch+1:3d}    {train_loss:.4f}    {val_loss:.4f}    {train_acc:6.2f}%    {val_acc:6.2f}%    {current_lr:.6f}")
        
        # Update visualizer
        visualizer.update(epoch + 1, train_loss, val_loss, train_acc, val_acc, current_lr)

        # Save best model
        if val_acc > best_acc:
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            best_acc = val_acc
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = f'mnist_model_{timestamp}_acc{val_acc:.2f}.pth'
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, best_model_path)

        if val_acc > 99.5:
            print("\nReached target accuracy. Stopping training.")
            break

    print("\nTraining completed. Best validation accuracy: {:.2f}%".format(best_acc))
    visualizer.plot_metrics()
    return best_acc

if __name__ == "__main__":
    for model_file in glob.glob('mnist_model_*.pth'):
        os.remove(model_file)
    train() 