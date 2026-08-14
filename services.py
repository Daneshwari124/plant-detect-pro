import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def analyze_plant_image(image_pil):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = [
        "Identify the plant and any diseases present. Provide the following in a structured format:",
        "1. Common Name of Plant",
        "2. Health Status (Healthy/Diseased)",
        "3. Disease Name (if any)",
        "4. Confidence Level (%)",
        "5. Short-term Treatment Plan",
        "6. Prevention Tips"
    ]
    
    response = model.generate_content([*prompt, image_pil])
    return response.text