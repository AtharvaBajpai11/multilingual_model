"""
IsoFlow-S2ST Studio Server
FastAPI Backend Server serving Video Upload, Multilingual IsoFlow Translation, 
and Synchronized Lip-Sync Video Streaming.
"""

import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from pipeline.video_processor import VideoProcessor
from pipeline.translator import MultilingualVideoTranslator, SUPPORTED_LANGUAGES
from pipeline.video_renderer import VideoRenderer
from pipeline.audio_translator import AudioTranslator

app = FastAPI(
    title="IsoFlow-S2ST Studio",
    description="Multilingual AI Video Translation & Isochronous Lip-Syncing Engine",
    version="1.0.0",
)

# Enable CORS for cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Pipeline instances
video_processor = VideoProcessor()
translator_engine = MultilingualVideoTranslator()
video_renderer = VideoRenderer(output_dir=OUTPUT_DIR)
audio_translator = AudioTranslator()


@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/languages")
def get_languages():
    """Returns dictionary of supported target translation languages."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/api/translate_audio")
async def translate_audio_endpoint(
    file: UploadFile = File(None),
):
    """
    Processes an uploaded audio file and translates its speech into English.
    """
    job_id = str(uuid.uuid4())[:8]

    if file and file.filename:
        file_ext = os.path.splitext(file.filename)[1] or ".wav"
        input_filename = f"upload_audio_{job_id}{file_ext}"
        input_path = os.path.join(UPLOAD_DIR, input_filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[Server] Saved uploaded audio to {input_path}")
    else:
        demo_path = os.path.join(UPLOAD_DIR, f"demo_audio_{job_id}.wav")
        AudioTranslator.generate_demo_audio(demo_path, duration_sec=3.5)
        input_path = demo_path
        print(f"[Server] Using generated demo audio at {input_path}")

    try:
        result = audio_translator.translate_to_english(audio_path=input_path)
        output_filename = f"translated_english_{job_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        audio_translator.save_audio(result["translated_waveform"], output_path)

        audio_url = f"/api/download/{output_filename}"

        return JSONResponse({
            "status": "success",
            "job_id": job_id,
            "target_language": "English (en-US)",
            "audio_url": audio_url,
            "metrics": result["metrics"],
        })
    except Exception as e:
        print(f"[Server] Audio translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/translate")
async def translate_video(
    file: UploadFile = File(None),
    target_lang: str = Form("es"),
    flow_steps: int = Form(10),
):
    """
    Processes uploaded video, executes IsoFlow-S2ST translation & lip-sync generation,
    and returns rendered video download URL.
    """
    job_id = str(uuid.uuid4())[:8]

    if file and file.filename:
        # Save uploaded video
        file_ext = os.path.splitext(file.filename)[1] or ".mp4"
        input_filename = f"upload_{job_id}{file_ext}"
        input_path = os.path.join(UPLOAD_DIR, input_filename)
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[Server] Saved uploaded video to {input_path}")
    else:
        # Use fallback sample video path
        input_path = os.path.join(UPLOAD_DIR, "sample_input.mp4")
        print(f"[Server] No file provided, processing default video trajectory.")

    try:
        # 1. Demux video into frames & audio spectrograms
        processed_inputs = video_processor.extract_audio_and_frames(input_path)

        # 2. IsoFlow-S2ST Translation & Duration-controlled Lip-Syncing
        translation_result = translator_engine.translate_and_sync(
            processed_inputs=processed_inputs,
            target_lang=target_lang,
            flow_steps=flow_steps,
        )

        # 3. Render final output video file
        output_filename = f"translated_{job_id}_{target_lang}.mp4"
        output_path = video_renderer.render_video(
            frames=translation_result["generated_frames"],
            audio_waveform=translation_result["target_audio_waveform"],
            output_filename=output_filename,
        )

        # Output video URL
        video_url = f"/api/download/{os.path.basename(output_path)}"

        return JSONResponse({
            "status": "success",
            "job_id": job_id,
            "target_language": translation_result["target_lang"],
            "video_url": video_url,
            "metrics": translation_result["metrics"],
        })

    except Exception as e:
        print(f"[Server] Error during translation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{filename}")
def download_file(filename: str):
    """Serves generated translated video and audio files."""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        if filename.endswith(".wav"):
            return FileResponse(file_path, media_type="audio/wav", filename=filename)
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    # Check if fallback GIF exists
    gif_path = file_path.replace(".mp4", ".gif")
    if os.path.exists(gif_path):
        return FileResponse(gif_path, media_type="image/gif", filename=os.path.basename(gif_path))
    
    raise HTTPException(status_code=404, detail="Requested file not found.")


if __name__ == "__main__":
    import uvicorn
    print("[Server] Starting IsoFlow-S2ST Studio at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
