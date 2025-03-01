from transformers import pipeline
sa = pipeline('text-classification', model='CAMeL-Lab/bert-base-arabic-camelbert-ca-sentiment')
sentences = ['أنا بخير', 'أنا لست بخير']
results = sa(sentences)
print(results)