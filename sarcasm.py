from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def detect_sarcasm(text):
    """Detects sarcasm in an Arabic text using the pretrained model."""
    model_name = "MohamedGalal/arabert-sarcasm-detector"
    
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
    # predictions = torch.argmax(outputs.logits, dim=1).item()
    # label_map = {0: "Not Sarcastic", 1: "Sarcastic"}
    
    return outputs

# Example usage with sarcastic text
text = "يا له من يوم رائع حقًا! الجو حار جدًا والكهرباء مقطوعة!"
result = detect_sarcasm(text)
print(f"Sarcasm Detection Result: {result}")
