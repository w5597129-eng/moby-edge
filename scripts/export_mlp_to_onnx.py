#!/usr/bin/env python3
"""
Export a saved PyTorch MLP checkpoint (dict or whole module) to ONNX.

Usage:
    python scripts/export_mlp_to_onnx.py [INPUT_PATH] [OUT_PATH]

Defaults:
    INPUT_PATH: models/mlp_classifier.pth
    OUT_PATH: models/mlp_classifier.onnx

The script attempts to:
- If the checkpoint is a dict containing 'model_state_dict' (+ optional input_size),
  construct the MLPClassifier defined in `src/inference_worker.py`, load the state_dict,
  and export with a dummy input shaped (1, input_size).
- If the checkpoint is a saved nn.Module, load it and export directly.
"""
import os
import sys
import torch
import argparse

DEFAULT_IN = os.path.join("models", "mlp_classifier.pth")
DEFAULT_OUT = os.path.join("models", "mlp_classifier.onnx")

parser = argparse.ArgumentParser()
parser.add_argument("input", nargs="?", default=DEFAULT_IN)
parser.add_argument("output", nargs="?", default=DEFAULT_OUT)
args = parser.parse_args()

inp = args.input
out = args.output

if not os.path.exists(inp):
    print(f"Input not found: {inp}")
    sys.exit(2)

# Local copy of simple MLP matching inference_worker.MLPClassifier
import torch.nn as nn
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64, 32), output_size=2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

ckpt = None
try:
    ckpt = torch.load(inp, map_location='cpu')
    print(f"Loaded torch object from {inp}")
except Exception as e:
    print(f"torch.load failed: {e}")
    sys.exit(1)

if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    input_size = ckpt.get('input_size')
    hidden_sizes = ckpt.get('hidden_sizes', (64,32))
    output_size = ckpt.get('output_size', 2)
    if input_size is None:
        # try to infer from state_dict
        for k,v in ckpt['model_state_dict'].items():
            if k.endswith('.weight') and v is not None:
                input_size = v.shape[1]
                break
    if input_size is None:
        print("Could not infer input_size from checkpoint; please provide a model with 'input_size' in checkpoint or a full module.")
        sys.exit(2)
    model = MLPClassifier(input_size, hidden_sizes, output_size)
    try:
        model.load_state_dict(ckpt['model_state_dict'])
    except Exception as e:
        print(f"Warning: load_state_dict failed: {e}")
else:
    # assume it's a module
    if isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        print("Checkpoint is not a dict with 'model_state_dict' and not an nn.Module. Cannot export.")
        sys.exit(2)

model.eval()
# Dummy input
# Try to infer input dim from model fc1 weight
try:
    w = None
    for name, param in model.named_parameters():
        if name.endswith('fc1.weight') or name.endswith('fc1.weight'):
            w = param
            break
    if w is not None:
        input_size = w.shape[1]
except Exception:
    pass

dummy = torch.randn(1, int(input_size))

try:
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
    )
    print(f"Exported ONNX model to {out}")
except Exception as e:
    print(f"ONNX export failed: {e}")
    sys.exit(1)

print("Done.")
