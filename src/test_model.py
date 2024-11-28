import torch
import pytest
from model import MNISTResNet
import io
import sys
from contextlib import redirect_stdout

def test_parameter_count():
    model = MNISTResNet()
    # Get direct parameter count first
    direct_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Also check through show_parameters()
    f = io.StringIO()
    with redirect_stdout(f):
        model.show_parameters()
    output = f.getvalue()
    
    # Extract total parameters from the output
    for line in output.split('\n'):
        if "Total Trainable Parameters:" in line:
            total_params = int(line.split(': ')[1].replace(',', ''))
            break
    
    # Test assertions
    assert direct_count < 20000, f"Model has {direct_count} parameters, exceeding limit of 20000"
    assert direct_count == total_params, f"Parameter count mismatch: direct {direct_count} vs reported {total_params}"
    print(f"Total parameters: {direct_count}")  # Debug print

def test_batch_norm():
    model = MNISTResNet()
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should include BatchNorm layers"

def test_dropout():
    model = MNISTResNet()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should include Dropout layers"

def test_output_shape():
    model = MNISTResNet()
    batch_size = 4  # Changed from 1 to 4 for better batch norm behavior
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    model.eval()  # Set to eval mode for testing
    with torch.no_grad():
        output = model(input_tensor)
    assert output.shape == (batch_size, 10), f"Expected output shape ({batch_size}, 10), got {output.shape}"

def test_model_training_mode():
    model = MNISTResNet()
    model.train()
    assert model.training, "Model should be in training mode"
    model.eval()
    assert not model.training, "Model should be in evaluation mode"

def test_forward_pass():
    model = MNISTResNet()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    model.eval()  # Set to eval mode for testing
    with torch.no_grad():
        output = model(input_tensor)
    assert not torch.isnan(output).any(), "Forward pass produced NaN values"
    assert not torch.isinf(output).any(), "Forward pass produced infinite values"

def test_show_parameters():
    model = MNISTResNet()
    f = io.StringIO()
    with redirect_stdout(f):
        model.show_parameters()
    output = f.getvalue()
    
    # Test that the output contains essential information
    required_sections = [
        "Model Parameter Details:",
        "Total Trainable Parameters:",
        "Layer-wise summary:"
    ]
    
    for section in required_sections:
        assert section in output, f"Missing section: {section}"
    
    # Verify parameter count is under limit
    for line in output.split('\n'):
        if "Total Trainable Parameters:" in line:
            params = int(line.split(': ')[1].replace(',', ''))
            assert params < 20000, f"Too many parameters: {params}" 