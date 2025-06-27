# backend/main.py

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

# --- 1. Initialize FastAPI App ---
app = FastAPI(title="Multimodal AI Backend")

# --- 2. Load Models and Processors ---
# This is done once when the app starts
# Use a smaller, CPU-friendly model
# Note: The first time you run this, it will download the model weights (a few GBs).

print("Loading models... This may take a moment.")

# For Image Captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# For Visual Question Answering (VQA)
vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Check for device (will use CPU if no GPU is available)
device = "cuda" if torch.cuda.is_available() else "cpu"
caption_model.to(device)
vqa_model.to(device)

print(f"Models loaded successfully on device: {device}")

# --- 3. Define Helper Function ---
def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Converts image bytes to a PIL Image."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

# --- 4. Define API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "Backend is running!"}

@app.post("/generate-caption")
async def generate_caption(image_file: UploadFile = File(...)):
    """
    Endpoint to generate a caption for an uploaded image.
    """
    image_bytes = await image_file.read()
    image = read_image_from_bytes(image_bytes)

    # Process image and generate caption
    inputs = caption_processor(images=image, return_tensors="pt").to(device)
    outputs = caption_model.generate(**inputs, max_new_tokens=50)
    caption = caption_processor.decode(outputs[0], skip_special_tokens=True)

    return {"caption": caption}

@app.post("/answer-question")
async def answer_question(question: str = Form(...), image_file: UploadFile = File(...)):
    """
    Endpoint to answer a question about an uploaded image.
    """
    image_bytes = await image_file.read()
    image = read_image_from_bytes(image_bytes)

    # Process image and question, then generate answer
    inputs = vqa_processor(images=image, text=question, return_tensors="pt").to(device)
    outputs = vqa_model.generate(**inputs, max_new_tokens=20)
    answer = vqa_processor.decode(outputs[0], skip_special_tokens=True)

    return {"answer": answer}