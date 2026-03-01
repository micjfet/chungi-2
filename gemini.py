from google import genai
import cv2
import PIL.Image
import os
import json

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_gemini_analysis(frame, telemetry_json):
    # 1. Parse the telemetry to find the language
    try:
        data = json.loads(telemetry_json)
        target_lang = data.get("target_language", "English")
    except Exception:
        target_lang = "English"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = PIL.Image.fromarray(frame_rgb)

    # 2. Update the prompt to enforce the language
    prompt = f"""
    You are a navigation assistant. Analyze this frame and the following telemetry:
    {telemetry_json}
    
    CRITICAL: You must provide your response entirely in {target_lang}.
    
    Give a concise, 1-sentence warning or instruction to the user based on the danger score and objects detected.
    Sound human in your response, and be specific in mentioning any objects that could be notable to the user.
    Don't mention the danger score itself, more of a general overview for a user.
    If there are no objects of note, give a more general overview of the frame.
    Assume any objects classified as refrigerators are instead walls.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[prompt, pil_img]
    )

    print(f"🤖 Gemini ({target_lang}): {response.text}")
    return response.text