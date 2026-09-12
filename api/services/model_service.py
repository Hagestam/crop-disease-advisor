"""
Handles TFLite model loading and image inference.
Loaded once on startup and shared across requests.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import io
import tflite_runtime.interpreter as tflite
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH       = Path("models/tflite/efficientnet_b0_int8.tflite")
CLASS_NAMES_PATH = Path("models/class_names.json")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 224


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class ModelService:
    def __init__(self):
        # Load class names
        with open(CLASS_NAMES_PATH) as f:
            self.class_names = json.load(f)

        # Load TFLite interpreter
        self.interpreter = tflite.Interpreter(model_path=str(MODEL_PATH))
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print(f"ModelService ready | {len(self.class_names)} classes")

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Decode, resize, and normalize an image for inference."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    def predict(self, image_bytes: bytes, top_k: int = 3) -> list[dict]:
        """
        Returns top-k predictions as a list of dicts:
        [{"class": str, "confidence": float, "crop": str, "disease": str}]
        """
        arr = self.preprocess(image_bytes)
        self.interpreter.set_tensor(self.input_details[0]["index"], arr)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        probs    = _softmax(logits)
        top_idxs = np.argsort(probs)[::-1][:top_k]

        results = []
        for idx in top_idxs:
            class_label = self.class_names[idx]
            parts  = class_label.split("___")
            crop   = parts[0].replace("_", " ")
            disease= parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
            results.append({
                "class":      class_label,
                "confidence": float(probs[idx]),
                "crop":       crop,
                "disease":    disease,
                "is_healthy": "healthy" in disease.lower(),
            })
        return results


# Singleton
_model_service: ModelService | None = None

def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service