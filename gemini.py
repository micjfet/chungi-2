from google import genai
import cv2
import PIL.Image
import os
import json

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_analysis(frame, telemetry_json):
    # Parse the telemetry to extract the language name
    try:
        data = json.loads(telemetry_json)
        target_lang = data.get("target_language", "English")
    except Exception:
        target_lang = "English"

    # Convert frame for Gemini
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(frame_rgb)

    prompt = f"""
    You are a navigation assistant for a person with visual impairments. 
    Analyze this frame and the following telemetry:
    {telemetry_json}
    
    CRITICAL: You must provide your response entirely in {target_lang}.
    
    Give a concise, 1-sentence warning or instruction to the user based on the danger score and objects detected.
    Sound human and helpful. Be specific about the location of objects (Left, Center, Right).
    Don't mention technical terms like "danger score" or "telemetry".
    If there are no objects of note, give a brief description of the path ahead.
    Assume any objects classified as "refrigerator" are actually "walls" or "large blockages".
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=[prompt, pil_img]
        )
        print(f"🤖 Gemini ({target_lang}): {response.text}")
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Error analyzing scene."