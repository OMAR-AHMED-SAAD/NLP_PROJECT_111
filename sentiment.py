from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def analyze_sentiment(text):
    """Analyzes sentiment in an Arabic text using the pretrained model."""
    model_name = "Walid-Ahmed/arabic-sentiment-model"
    
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get prediction
    probabilities = torch.softmax(outputs.logits, dim=1)
    positive_probability = probabilities[0][1].item()
    
    return positive_probability

# Example usage with sentiment analysis
text = "هذا نص جيد"
result = analyze_sentiment(text)
print(f"Sentiment Analysis Result: {result}")