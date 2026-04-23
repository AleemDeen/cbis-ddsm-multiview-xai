"""
FastAPI server for the Mammogram XAI application.

Exposes two endpoints:
  GET  /models   — list all available .pt checkpoints
  POST /predict  — run inference on uploaded DICOM file(s) and return overlays

Run from the project root:
    uvicorn src.api.server:app --reload --port 8000
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.model_runner import list_available_models, run_inference

app = FastAPI(title="Mammogram XAI API", version="1.0.0")

# Allow requests from the Vite development server and standard localhost ports.
# Without CORS headers the browser blocks cross-origin requests from the
# frontend (port 5173) to the API (port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Simple liveness check — confirms the server is running."""
    return {"status": "ok"}


@app.get("/models")
def get_models():
    """Return a list of all .pt model files found in the models/ directory."""
    models = list_available_models()
    return {"models": models}


@app.post("/predict")
async def predict(
    model_name: str        = Form(...),
    cc_file:    UploadFile = File(...),
    mlo_file:   UploadFile = File(None),   # optional — only required for multi-view models
):
    """
    Run inference on one or two DICOM files and return base64-encoded overlay images.

    The model type is inferred automatically from the filename (sv_* = single-view,
    mv_best.pt = segmentation, mv_* = multi-view GradCAM). The response includes
    the original mammogram and the appropriate XAI overlay for each view.
    """
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
