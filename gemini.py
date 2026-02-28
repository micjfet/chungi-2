from google import genai

client = genai.Client(api_key="AIzaSyD_LNZHMHQOchtewbcTnqeCy0Eo2vSdZDk")

response = client.models.generate_content(
    model="gemini-2.5-flash-lite", contents="Explain how AI works in a few words"
)
print(response.text)