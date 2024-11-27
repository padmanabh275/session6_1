import torch
import pytest
from model import MNISTNet

@pytest.fixture
def model():
    return MNISTNet()

def test_parameter_count(model):
    param_count = model.count_parameters()
    assert param_count < 20000, f"Model has {param_count} parameters, should be less than 20000"

def test_batch_norm_layers(model):
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use batch normalization"

def test_dropout_layers(model):
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use dropout"

def test_gap_layer(model):
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap, "Model should use Global Average Pooling"

def test_input_output_shape(model):
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (batch_size, 10), f"Expected output shape (4, 10), got {output.shape}" 