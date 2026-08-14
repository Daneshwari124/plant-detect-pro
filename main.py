import io
import os
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=API_KEY)

# Use a current multimodal Gemini model.
MODEL_NAME = "gemini-3.6-flash"


@app.get("/")
async def health():
    return {"status": "Online"}


@app.post("/detect")
async def detect_disease(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="No image was uploaded."
            )

        # Open image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image."
            )

        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize large phone photos
        image.thumbnail((1024, 1024))

        prompt = """
Act as an expert plant pathologist.

Examine the provided image carefully.

1. Identify the plant species if possible.
2. Determine whether the leaf/plant appears healthy.
3. If there is a disease, pest, fungal infection, bacterial infection,
   nutrient deficiency, or other visible problem, identify the most
   likely condition.
4. Give a confidence percentage.
5. Provide practical treatment and prevention steps.
6. If the image is not a plant, set condition to "Not a plant".
7. Do not invent a disease if the evidence is unclear.

Return ONLY valid JSON in exactly this structure:

{
  "name": "Plant Name",
  "condition": "Disease Name, Healthy, or Not a plant",
  "confidence": "90%",
  "treatment": "Detailed treatment and prevention steps"
}

Do not use markdown.
Do not put the JSON inside ```json.
"""

        print(f"Sending image to Gemini model: {MODEL_NAME}")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                prompt,
                image,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=1,
                top_k=1,
                max_output_tokens=2048,
                response_mime_type="application/json",
            ),
        )

        text = response.text.strip()

        print("Gemini response:")
        print(text)

        if not text:
            raise Exception("Gemini returned an empty response.")

        # Remove markdown fences just in case
        if text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

        result = json.loads(text)

        # Make sure expected fields exist
        result.setdefault("name", "Unknown")
        result.setdefault("condition", "Unknown")
        result.setdefault("confidence", "Unknown")
        result.setdefault("treatment", "No treatment information available.")

        return result

    except HTTPException:
        raise

    except Exception as e:
        print("========== GEMINI ERROR ==========")
        print(repr(e))
        print("===================================")

        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )