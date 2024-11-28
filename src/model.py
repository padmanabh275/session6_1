import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

dropout_value = 0.08

class MNISTResNet(nn.Module):
    def __init__(self):
        super(MNISTResNet, self).__init__()
        
        # First block - 3 convolutions with BatchNorm and Dropout
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.drop1 = nn.Dropout(dropout_value)
        
        self.conv2 = nn.Conv2d(8, 8, 3)
        self.bn2 = nn.BatchNorm2d(8)
        self.drop2 = nn.Dropout(dropout_value)
        
        self.conv3 = nn.Conv2d(8, 8, 3)
        self.bn3 = nn.BatchNorm2d(8)
        self.drop3 = nn.Dropout(dropout_value)
        
        # MaxPool and 1x1 conv
        self.pool1 = nn.MaxPool2d(2, 2)
        self.onecross = nn.Conv2d(8, 16, 1)
        
        # Second block - 2 convolutions
        self.conv4 = nn.Conv2d(16, 16, 3)
        self.bn4 = nn.BatchNorm2d(16)
        self.drop4 = nn.Dropout(dropout_value)
        
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.bn5 = nn.BatchNorm2d(16)
        self.drop5 = nn.Dropout(dropout_value)
        
        # Final convolution
        self.conv6 = nn.Conv2d(16, 10, 7)

    def forward(self, x):
        # First block with skip connection
        x1 = self.drop1(self.bn1(F.relu(self.conv1(x))))
        x2 = self.drop2(self.bn2(F.relu(self.conv2(x1))))
        x3 = self.drop3(self.bn3(F.relu(self.conv3(x2))))
        
        # MaxPool and 1x1 conv
        x = self.onecross(self.pool1(x3))
        
        # Second block with skip connection
        x4 = self.drop4(self.bn4(F.relu(self.conv4(x))))
        x5 = self.drop5(self.bn5(F.relu(self.conv5(x4))))
        
        # Final conv
        x = self.conv6(x5)
        
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def show_parameters(self):
        """Display model parameters and structure"""
        total_params = self.count_parameters()
        
        print("\nModel Parameter Details:")
        print("------------------------")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.numel():,} parameters")
        
        print(f"\nTotal Trainable Parameters: {total_params:,}")
        
        print("\nLayer-wise summary:")
        print("-" * 80)
        print(f"{'Layer':<40} {'Output Shape':<20} {'Param #':<10}")
        print("-" * 80)
        
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
                    shape = str(tuple(module.weight.shape))
                else:
                    shape = "N/A"
                print(f"{name:<40} {shape:<20} {params:<10,d}")
        
        print("-" * 80)

if __name__ == "__main__":
    model = MNISTResNet()
    model.show_parameters() 