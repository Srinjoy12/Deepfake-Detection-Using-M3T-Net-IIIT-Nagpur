import gradio as gr
import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from model import get_model, sliding_window_inference
import json

# Load the model on startup
MODEL_PATH = "visual_only_best_model.pth"
model = None

def load_model():
    global model
    model = get_model(MODEL_PATH)
    return "Model loaded successfully!"

def analyze_video(video_file):
    """Analyze video for deepfake detection"""
    if video_file is None:
        return "Please upload a video file", None, None
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            temp_path = tmp_file.name
        
        # Run analysis
        results = []
        final_result = None
        
        for message in sliding_window_inference(model, temp_path):
            if message.startswith("LOG:"):
                results.append(message[4:])  # Remove "LOG:" prefix
            elif message.startswith("RESULT:"):
                result_json = message[7:]  # Remove "RESULT:" prefix
                final_result = json.loads(result_json)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if final_result:
            status = "🚨 DEEPFAKE DETECTED" if final_result['is_deepfake'] else "✅ AUTHENTIC CONTENT"
            confidence = f"Confidence: {final_result['confidence']:.1%}"
            details = f"""
**Analysis Results:**
- Status: {status}
- {confidence}
- Total Frames: {final_result.get('total_frames', 'N/A')}
- Duration: {final_result.get('video_duration_seconds', 0):.1f}s
- Windows Analyzed: {final_result.get('windows_analyzed', 'N/A')}
            """
            
            return details, "\n".join(results), final_result
        else:
            return "Analysis failed - no results", "\n".join(results), None
            
    except Exception as e:
        return f"Error: {str(e)}", "", None

# Create Gradio interface
def create_gradio_app():
    with gr.Blocks(title="🔍 Deepfake Detector", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🔍 Deepfake Detector
        
        Upload a video file to analyze it for deepfake content using advanced AI detection.
        
        **⚠️ Disclaimer:** Results are not guaranteed to be 100% accurate and should be used for informational purposes only.
        """)
        
        with gr.Row():
            with gr.Column():
                video_input = gr.File(
                    label="📹 Upload Video",
                    file_types=[".mp4", ".mov", ".avi", ".mkv"],
                    file_count="single"
                )
                
                analyze_btn = gr.Button("🔍 Analyze Video", variant="primary", size="lg")
                
            with gr.Column():
                result_output = gr.Markdown(label="📊 Analysis Results")
                
        with gr.Row():
            logs_output = gr.Textbox(
                label="📝 Analysis Logs",
                lines=10,
                max_lines=20,
                show_copy_button=True
            )
        
        # Load model on startup
        gr.Markdown("### 🤖 Model Status")
        model_status = gr.Textbox(value="Loading model...", label="Status")
        
        app.load(load_model, outputs=model_status)
        
        analyze_btn.click(
            analyze_video,
            inputs=[video_input],
            outputs=[result_output, logs_output, gr.State()]
        )
        
        gr.Markdown("""
        ### 🛠️ Technical Details
        - **Model**: M2T2-Net (Visual-only)
        - **Framework**: PyTorch + FastAPI
        - **Processing**: Sliding window analysis
        - **Inference**: Real-time on GPU
        """)
    
    return app

# Create FastAPI app for API access
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = get_model(MODEL_PATH)
    yield
    model = None

fastapi_app = FastAPI(lifespan=lifespan)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hugging Face Spaces needs open CORS
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.post("/analyze/")
async def analyze_endpoint(file: UploadFile = File(...)):
    """FastAPI endpoint for programmatic access"""
    # Implementation similar to your existing main.py
    pass

# For Hugging Face Spaces
if __name__ == "__main__":
    # Check if running on Hugging Face Spaces
    if os.getenv("SPACE_ID"):
        # Running on HF Spaces - use Gradio
        app = create_gradio_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
    else:
        # Running locally - use FastAPI
        import uvicorn
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000) 