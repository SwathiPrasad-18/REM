
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from REM_Inference.self_learning_bridge import infer_and_queue

app = FastAPI(
    title="REM Cyber Threat Intelligence API",
    description=(
        "API for REM hybrid threat inference and "
        "review-queue integration."
    ),
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    sample_id: str = Field(
        min_length=1,
        max_length=200,
        description="Unique identifier for this sample",
    )
    api_calls: List[str] = Field(
        min_length=1,
        max_length=10000,
        description="Ordered API-call sequence",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "REM Inference API",
        "training_enabled": False,
        "autonomous_response_enabled": False,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return infer_and_queue(
            api_calls=request.api_calls,
            sample_id=request.sample_id,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"REM inference failed: {type(exc).__name__}",
        ) from exc
