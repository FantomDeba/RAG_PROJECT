import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

print("Available models:")
for m in genai.list_models():
    print(m.name)