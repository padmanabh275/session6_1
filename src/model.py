import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

dropout_value = 0.015

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        mid_channels = max(in_channels, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_value)
        )

    def forward(self, x):
        return self.conv(x)

class SEBlock(nn.Module):
    def __init__(self, channels):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels, bias=False),
            nn.ReLU(),
            nn.Linear(channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x

class MNISTResNet(nn.Module):
    def __init__(self):
        super(MNISTResNet, self).__init__()
        
        # Initial Convolution Block
        self.convblock1 = ConvBlock(1, 16, kernel_size=3, padding=1)  # 28x28x16
        self.se1 = SEBlock(16)
        self.convblock2 = ConvBlock(16, 16, kernel_size=3, padding=1)  # 28x28x16
        
        # Transition Block 1
        self.transition1 = TransitionBlock(16, 24)  # 14x14x24
        
        # Convolution Block 2
        self.convblock3 = ConvBlock(24, 28, kernel_size=3, padding=1)  # 14x14x28
        self.se2 = SEBlock(28)
        self.convblock4 = ConvBlock(28, 28, kernel_size=3, padding=1)  # 14x14x28
        
        # Transition Block 2
        self.transition2 = TransitionBlock(28, 32)  # 7x7x32
        
        # Convolution Block 3
        self.convblock5 = ConvBlock(32, 36, kernel_size=3, padding=1)  # 7x7x36
        self.se3 = SEBlock(36)
        self.convblock6 = ConvBlock(36, 40, kernel_size=3, padding=0)  # 5x5x40
        self.convblock7 = ConvBlock(40, 48, kernel_size=3, padding=0)  # 3x3x48
        
        # Global Average Pooling and Max Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        
        # Final 1x1 convolution
        self.convblock8 = nn.Sequential(
            nn.Conv2d(48 * 2, 32, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 10, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # Block 1 with skip connection and SE
        x1 = self.convblock1(x)
        x1 = self.se1(x1)
        x = x1 + self.convblock2(x1)  # Skip connection
        
        # Transition 1
        x = self.transition1(x)
        
        # Block 2 with skip connection and SE
        x2 = self.convblock3(x)
        x2 = self.se2(x2)
        x = x2 + self.convblock4(x2)  # Skip connection
        
        # Transition 2
        x = self.transition2(x)
        
        # Block 3 with SE
        x = self.convblock5(x)
        x = self.se3(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        
        # Parallel pooling
        avg_x = self.gap(x)
        max_x = self.gmp(x)
        x = torch.cat([avg_x, max_x], dim=1)
        
        # Final conv
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

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
        
        print("\nTotal Trainable Parameters:", total_params)
        
        try:
            # Try to create a sample input tensor
            device = next(self.parameters()).device
            batch_size = 1
            sample_input = torch.randn(batch_size, 1, 28, 28).to(device)
            
            # Get output shape
            with torch.no_grad():
                output = self(sample_input)
            
            print("\nInput shape:", (batch_size, 1, 28, 28))
            print("Output shape:", tuple(output.shape))
            
            # Print layer-wise summary
            print("\nLayer-wise summary:")
            print("-" * 80)
            print(f"{'Layer':<40} {'Output Shape':<20} {'Param #':<10}")
            print("-" * 80)
            
            total = 0
            for name, module in self.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    total += params
                    if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
                        shape = str(tuple(module.weight.shape))
                    else:
                        shape = "N/A"
                    print(f"{name:<40} {shape:<20} {params:<10,d}")
            
            print("-" * 80)
            print(f"Total trainable parameters: {total:,}")
            
        except Exception as e:
            print("\nCould not generate detailed summary:", str(e))
            print("Basic parameter count:", total_params)

if __name__ == "__main__":
    model = MNISTResNet()
    model.show_parameters() 