import io
import os
import json
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
# Free-tier quota is counted per model, so if one model runs out for the day
# the next one in this list is tried. Override on Render with an environment
# variable GEMINI_MODELS="modelA,modelB,modelC" (no code change needed).
MODEL_NAMES = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODELS",
        "gemini-3.6-flash,gemini-3.5-flash,gemini-3.1-flash-lite",
    ).split(",")
    if m.strip()
]

# Languages the app can ask for (code -> name used in the prompt)
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
}


@app.get("/")
async def health():
    return {"status": "Online"}


@app.post("/detect")
async def detect_disease(file: UploadFile = File(...), lang: str = Form("en")):
    lang = lang if lang in LANGUAGES else "en"
    language_name = LANGUAGES[lang]
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

        prompt = f"""
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
8. Write the "condition" and "treatment" values in simple, clear
   {language_name} that a farmer with little schooling can understand.
   Keep the JSON keys in English. Write the "name" value in {language_name}
   (you may add the English name in brackets).

Return ONLY valid JSON in exactly this structure:

{{
  "name": "Plant Name",
  "condition": "Disease Name, Healthy, or Not a plant",
  "confidence": "90%",
  "treatment": "Detailed treatment and prevention steps"
}}

Do not use markdown.
Do not put the JSON inside ```json.
"""


        response = None
        quota_hit = False
        busy_hit = False
        last_error = None
        for model_name in MODEL_NAMES:
            # Up to 2 tries per model (a short wait helps with "high demand")
            for attempt in range(2):
                try:
                    print(f"Sending image to Gemini model: {model_name} (try {attempt + 1})")
                    response = client.models.generate_content(
                        model=model_name,
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
                    break
                except Exception as e:
                    last_error = e
                    msg = str(e)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        quota_hit = True
                        print(f"Quota exhausted for {model_name}, trying next model")
                        break
                    if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower():
                        busy_hit = True
                        print(f"{model_name} is busy (503)")
                        if attempt == 0:
                            time.sleep(2)
                            continue
                        break
                    if "404" in msg or "NOT_FOUND" in msg:
                        print(f"Model {model_name} not available, trying next model")
                        break
                    raise
            if response is not None:
                break

        if response is None:
            if busy_hit:
                raise HTTPException(
                    status_code=503,
                    detail="AI is very busy right now. Please try again in a minute.",
                )
            if quota_hit:
                raise HTTPException(
                    status_code=429,
                    detail="Daily AI limit reached. Please try again later.",
                )
            raise last_error or Exception("No Gemini model available.")

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

        result["lang"] = lang
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