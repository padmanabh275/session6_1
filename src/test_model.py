import torch
import pytest
from model import MNISTResNet

def test_parameter_count():
    model = MNISTResNet()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params < 20000, f"Model has {total_params} parameters, exceeding limit of 20000"

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
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), f"Expected output shape (1, 10), got {output.shape}"

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
    output = model(input_tensor)
    assert not torch.isnan(output).any(), "Forward pass produced NaN values"
    assert not torch.isinf(output).any(), "Forward pass produced infinite values" 