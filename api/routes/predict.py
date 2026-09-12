from fastapi import APIRouter, File, UploadFile, HTTPException
from api.services.model_service import get_model_service

router = APIRouter(prefix="/predict", tags=["Detection"])


@router.post("/")
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload a crop image and receive disease detection predictions.
    Returns top-3 predictions with confidence scores.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png)")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 10MB)")

    service = get_model_service()
    predictions = service.predict(image_bytes, top_k=3)

    return {
        "predictions":  predictions,
        "top_prediction": predictions[0],
    }