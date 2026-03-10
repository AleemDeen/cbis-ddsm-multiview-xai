"""
FastAPI server for the Mammogram XAI application.

Run from project root:
    uvicorn src.api.server:app --reload --port 8000
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.model_runner import list_available_models, run_inference

app = FastAPI(title="Mammogram XAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    models = list_available_models()
    return {"models": models}


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    cc_file:    UploadFile = File(...),
    mlo_file:   UploadFile = File(None),
):
    if not model_name.endswith(".pt"):
        raise HTTPException(400, "model_name must be a .pt filename")

    try:
        cc_bytes  = await cc_file.read()
        mlo_bytes = await mlo_file.read() if mlo_file else None
        result    = run_inference(model_name, cc_bytes, mlo_bytes)
        return JSONResponse(content=result)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")
