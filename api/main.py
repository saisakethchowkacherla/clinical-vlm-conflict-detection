from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .inference import run_inference
from .validation import validate_xray_image

app = FastAPI(title="Clinical VLM Conflict Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

CORS_HEADERS = {"Access-Control-Allow-Origin": "*"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"error": True, "message": f"Internal server error: {exc}"},
        headers=CORS_HEADERS,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    content = detail if isinstance(detail, dict) else {"error": True, "message": detail}
    return JSONResponse(status_code=exc.status_code, content=content, headers=CORS_HEADERS)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Chest X-ray image"),
    lab_value: float = Form(..., description="Lab value (e.g. WBC count)"),
    ref_low: float = Form(..., description="Reference range lower bound"),
    ref_high: float = Form(..., description="Reference range upper bound"),
) -> dict[str, object]:
    if ref_low >= ref_high:
        raise HTTPException(status_code=422, detail="ref_low must be less than ref_high")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty")

    mime = image.content_type or "image/png"
    validate_xray_image(image_bytes, image.filename or "", mime)

    result = run_inference(image_bytes, lab_value, ref_low, ref_high, mime)

    return {
        "image_only_prediction":  result.image_only_prediction,
        "multimodal_prediction":  result.multimodal_prediction,
        "conflict_detected":      result.conflict_detected,
        "defer_gate":             result.defer_gate,
        "lab_value":              result.lab_value,
        "lab_status":             result.lab_status,
        "validation_passed":      True,
        "image_type":             result.image_type,
        "image_type_confidence":  result.image_type_confidence,
        "model": {
            "name":                  "densenet121-res224-all",
            "composite_score":       result.composite_score,
            "image_only_threshold":  result.image_only_threshold,
            "multimodal_threshold":  result.multimodal_threshold,
            "pathology_scores":      result.model_scores,
        },
    }
