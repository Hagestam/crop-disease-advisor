"""
Converts PyTorch EfficientNet-B0 to INT8 TFLite model.
Pipeline: PyTorch → ONNX → TF SavedModel → TFLite INT8
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from pathlib import Path
from PIL import Image

CHECKPOINT_PATH  = Path("models/checkpoints/best_efficientnet_b0.pth")
ONNX_PATH        = Path("models/efficientnet_b0.onnx")
TF_SAVEDMODEL_DIR= Path("models/tf_savedmodel")
TFLITE_PATH      = Path("models/tflite/efficientnet_b0_int8.tflite")
CALIB_DIR        = Path("data/calibration")
IMG_SIZE         = 224
NUM_CLASSES      = 38

TFLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── STEP 1: Load PyTorch model ───────────────────────────────────────────────
print("[1/4] Loading PyTorch model...")
device = torch.device("cpu")  # Export on CPU for stability
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

model = efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, NUM_CLASSES),
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("  ✓ Model loaded")

# ─── STEP 2: Export to ONNX ───────────────────────────────────────────────────
print("[2/4] Exporting to ONNX...")
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_PATH),
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
)

# Verify the ONNX model
import onnx
onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)
print(f"  ✓ ONNX model saved to {ONNX_PATH}")

# ─── STEP 3: Convert ONNX → TF SavedModel ────────────────────────────────────
# We use onnx2tf for this conversion
print("[3/4] Converting ONNX → TF SavedModel (this may take a few minutes)...")
import subprocess
result = subprocess.run([
    "onnx2tf",
    "-i", str(ONNX_PATH),
    "-o", str(TF_SAVEDMODEL_DIR),
    "--non_verbose",
], capture_output=True, text=True)

if result.returncode != 0:
    print("  STDERR:", result.stderr)
    raise RuntimeError("onnx2tf conversion failed")
print(f"  ✓ TF SavedModel saved to {TF_SAVEDMODEL_DIR}")

# ─── STEP 4: Convert TF SavedModel → TFLite INT8 ─────────────────────────────
print("[4/4] Converting to TFLite INT8...")

import tensorflow as tf

# Build representative dataset from calibration images
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

calib_images = list(CALIB_DIR.glob("*.jpg")) + list(CALIB_DIR.glob("*.png"))
print(f"  Using {len(calib_images)} calibration images")

def representative_dataset():
    for img_path in calib_images:
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = (arr - MEAN) / STD
            arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
            yield [arr]
        except Exception as e:
            print(f"  Warning: skipping {img_path}: {e}")
            continue

# TFLite converter with INT8 full quantization
converter = tf.lite.TFLiteConverter.from_saved_model(str(TF_SAVEDMODEL_DIR))
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,  # fallback for ops not supported in INT8
]
converter.inference_input_type  = tf.float32  # Keep float32 I/O for Streamlit ease
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

# Size comparison
original_size = os.path.getsize(ONNX_PATH) / 1e6
tflite_size   = os.path.getsize(TFLITE_PATH) / 1e6
reduction     = (1 - tflite_size / original_size) * 100

print(f"\n  ✓ TFLite model saved to {TFLITE_PATH}")
print(f"  Original ONNX size:   {original_size:.1f} MB")
print(f"  TFLite INT8 size:     {tflite_size:.1f} MB")
print(f"  Size reduction:       {reduction:.1f}%")