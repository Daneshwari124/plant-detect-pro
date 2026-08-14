import io
import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI — set GEMINI_API_KEY as an environment variable
# (in your .env file locally, or in Render's dashboard for deployment).
# No hardcoded fallback key: never ship a live key in source code.
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it in your .env file (local) or in Render's Environment settings (deployed)."
    )

genai.configure(api_key=API_KEY)

# Cache the candidate model list so we only hit the Gemini "list models"
# endpoint once (at startup / first request), not on every /detect call.
_CANDIDATE_MODELS_CACHE = None
# Once a model succeeds, remember it and try it first on future requests
# so we skip straight to a known-good model instead of re-trying the
# whole fallback list every time.
_LAST_GOOD_MODEL = None

def get_candidate_models():
    """Get all models that support image analysis, sorted by stability.
    Result is cached in-memory for the lifetime of the server process."""
    global _CANDIDATE_MODELS_CACHE
    if _CANDIDATE_MODELS_CACHE is not None:
        candidates = _CANDIDATE_MODELS_CACHE
    else:
        try:
            all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            priority_keywords = ["1.5-flash", "flash-latest", "2.0-flash", "2.5-flash", "pro"]

            sorted_candidates = []
            for kw in priority_keywords:
                for m in all_models:
                    if kw in m and m not in sorted_candidates:
                        sorted_candidates.append(m)

            for m in all_models:
                if m not in sorted_candidates:
                    sorted_candidates.append(m)

            candidates = sorted_candidates
        except Exception as e:
            print(f"Error listing models: {e}")
            candidates = ["models/gemini-1.5-flash"]

        _CANDIDATE_MODELS_CACHE = candidates

    # Put the last model that actually worked at the front of the line.
    if _LAST_GOOD_MODEL and _LAST_GOOD_MODEL in candidates:
        candidates = [_LAST_GOOD_MODEL] + [m for m in candidates if m != _LAST_GOOD_MODEL]

    return candidates

@app.get("/")
async def health():
    return {"status": "Online"}

@app.post("/detect")
async def detect_disease(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Downscale large phone photos before sending to Gemini — this
        # cuts upload time noticeably and doesn't hurt diagnosis quality.
        image.thumbnail((1024, 1024))

        candidates = get_candidate_models()
        last_error = ""

        # Set generation config to minimize randomness and improve consistency
        generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
        }

        for model_name in candidates:
            try:
                print(f"Trying AI Model: {model_name}")
                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config
                )
                
                prompt = """Act as an expert plant pathologist.
                Examine the image carefully.
                1. Identify the specific plant species.
                2. If it is a leaf, diagnose any disease, pest, or nutrient issue.
                3. If healthy, state 'Healthy'.
                4. Provide detailed treatment/prevention.
                5. If NOT a plant, set condition to 'Not a plant'.

                Return ONLY a JSON object:
                {
                  "name": "Plant Name",
                  "condition": "Disease Name or Healthy",
                  "confidence": "90%",
                  "treatment": "Detailed steps"
                }"""
                
                response = model.generate_content([prompt, image])
                
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                global _LAST_GOOD_MODEL
                _LAST_GOOD_MODEL = model_name

                return json.loads(text)

            except Exception as e:
                last_error = str(e)
                print(f"Model {model_name} failed: {last_error}")
                if "429" in last_error or "404" in last_error or "Quota" in last_error:
                    continue
                else:
                    break

        raise HTTPException(
            status_code=500, 
            detail=f"AI error: {last_error}"
        )

    except Exception as e:
        print(f"Fatal Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Render (and most hosts) assign the port dynamically via $PORT.
    # Falls back to 8000 for local development.
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)