import google.generativeai as genai

def analyze_sarcasm(text):
    """Analyzes sarcasm in an Arabic text using Gemini Flash 2.0 API."""
    
    # Configure Gemini Flash 2.0 API
    genai.configure(api_key="AIzaSyB-C-HkY-PKqlj1zwkWchO3NqAkNy5E9hs")
    
    model = genai.GenerativeModel("gemini-flash-2")
    
    # Generate response
    response = model.generate_content(f"Detect sarcasm in the following Arabic text: '{text}'. Respond with 'Sarcastic' or 'Not Sarcastic'.")
    
    return response.text

# Example usage with sarcasm analysis
text = "يا له من يوم رائع حقًا! الجو حار جدًا والكهرباء مقطوعة!"
result = analyze_sarcasm(text)
print(f"Sarcasm Detection Result: {result}")
