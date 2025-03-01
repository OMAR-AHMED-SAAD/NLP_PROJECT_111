from transformers import pipeline
sarcasm_detector = pipeline("text-classification", model="MohamedGalal/arabert-sarcasm-detector")
text = "أوه، بالطبع، لأن الحياة دائمًا سهلة ومليئة بالورود، أليس كذلك؟"
result = sarcasm_detector(text)
print(result)
