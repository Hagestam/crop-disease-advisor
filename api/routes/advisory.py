from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from api.services.model_service import get_model_service
from rag.pipeline import get_rag

router = APIRouter(prefix="/advisory", tags=["Advisory"])


class AdvisoryRequest(BaseModel):
    crop: str
    disease: str


@router.post("/")
async def get_advisory(req: AdvisoryRequest):
    """Get treatment advisory for a given crop + disease."""
    if req.disease.lower() == "healthy":
        return {
            "advisory": f"✅ Your {req.crop} plant appears healthy! "
                        f"Continue regular monitoring, maintain proper spacing for airflow, "
                        f"and inspect leaves weekly for any early signs of disease.",
            "sources": [],
        }

    rag = get_rag()
    result = rag.get_advisory(crop=req.crop, disease=req.disease)
    return result


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    All-in-one endpoint: detect disease from image + return advisory.
    This is the primary endpoint used by the Streamlit frontend.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    
    # Step 1: Detect disease
    model_svc = get_model_service()
    predictions = model_svc.predict(image_bytes, top_k=3)
    top = predictions[0]

    # Step 2: Get advisory
    rag = get_rag()
    if top["is_healthy"]:
        advisory_result = {
            "advisory": f"✅ Your {top['crop']} plant appears healthy! "
                        f"Keep monitoring and maintain good agricultural practices.",
            "sources": [],
        }
    else:
        advisory_result = rag.get_advisory(crop=top["crop"], disease=top["disease"])

    return {
        "predictions":    predictions,
        "top_prediction": top,
        "advisory":       advisory_result["advisory"],
        "sources":        advisory_result["sources"],
    }