"""
FastAPI application entry point.
Start with: uvicorn api.main:app --reload --port 8000
"""

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes.predict  import router as predict_router
from api.routes.advisory import router as advisory_router

app = FastAPI(
    title="Crop Disease Detection & Advisory API",
    description=(
        "Detects plant diseases from images using EfficientNet-B0 (INT8 TFLite) "
        "and generates treatment recommendations via RAG pipeline."
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",
)

# CORS (allow Streamlit at localhost:8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - t0)*1000:.1f}ms"
    return response

# Include routers
app.include_router(predict_router)
app.include_router(advisory_router)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "message": "Crop Disease Detection API"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}


# Warm up models on startup
@app.on_event("startup")
async def startup_event():
    print("Warming up model service...")
    from api.services.model_service import get_model_service
    get_model_service()
    print("Warming up RAG pipeline...")
    from rag.pipeline import get_rag
    get_rag()
    print("API ready.")