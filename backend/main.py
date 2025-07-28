from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
import shutil
import os
import asyncio
from contextlib import asynccontextmanager
from model import get_model, sliding_window_inference
from report_generator import generate_report
from pydantic import BaseModel
from typing import List, Optional

# Load the model on startup
MODEL_PATH = "visual_only_best_model.pth"
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global model
    model = get_model(MODEL_PATH)
    yield
    # Clean up the model and release the resources
    model = None

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3002", "https://*.vercel.app"],  # Allow local and Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

async def run_analysis(file_path: str):
    """
    Generator function that runs the analysis and yields SSE-formatted messages.
    """
    for message in sliding_window_inference(model, file_path):
        print(message)
        yield f"data: {message}\n\n"
        await asyncio.sleep(0.05)  # Small delay to allow messages to be sent

@app.post("/analyze/")
async def analyze_file(file: UploadFile = File(...)):
    """
    Accepts a file, saves it, and streams the analysis progress.
    """
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Wrap the generator in a streaming response
    response = StreamingResponse(run_analysis(file_path), media_type="text/event-stream")

    # The file should be cleaned up after the stream is finished.
    # FastAPI does not have a direct post-response hook, so this is a bit tricky.
    # For this example, we'll assume the client closes the connection and we don't clean up,
    # or we accept that a lot of files will be stored temporarily.
    # A more robust solution would involve a background task.

    return response

class AnalysisResult(BaseModel):
    filename: str
    is_deepfake: bool
    confidence: float
    probabilities: List[float]
    face_images_b64: List[str] = []
    total_frames: Optional[int] = None
    video_duration_seconds: Optional[float] = None
    windows_analyzed: Optional[int] = None

@app.post("/generate-report/")
async def generate_report_endpoint(result: AnalysisResult):
    pdf_bytes = bytes(generate_report(result.dict()))
    
    headers = {
        'Content-Disposition': 'attachment; filename="deepfake_report.pdf"'
    }
    return Response(content=pdf_bytes, media_type='application/pdf', headers=headers)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 