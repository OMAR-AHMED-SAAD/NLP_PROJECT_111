import re
import json
from .bpe_tokenizer import BPETokenizer
from .glove_tokenizer import GloveTokenizer
from .qa_dataset import QADataset
from .qa_dataset_glove import QADatasetGlove
from torch.utils.data import DataLoader


def clean_text(text: str, is_question: bool=False):
    """
    Cleans text by removing extra spaces, newline characters, and special symbols.
    """
    if is_question:
        text = text.replace("\n", " ").replace("\t", " ")  # Remove newlines & tabs
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        text = text.strip()  # Trim leading/trailing spaces
    text = text.lower()  # Convert to lowercase
    return text



def load_and_process_squad(filepath, max_samples=20000):
    """
    Loads, cleans, and extracts answerable questions from the SQuAD dataset.

    Args:
        filepath (str): Path to the SQuAD JSON file.
        max_samples (int): Maximum number of answerable questions to load.

    Returns:
        List[dict]: A list of cleaned question-answer pairs.
    """
    with open(filepath, "r") as f:
        squad_data = json.load(f)

    data = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            context = clean_text(paragraph["context"]) 
            for qa in paragraph["qas"]:
                if qa["is_impossible"]:  
                    continue
                
                question = clean_text(qa["question"], is_question=True) 
                answer_text = clean_text(qa["answers"][0]["text"])  
                answer_start = qa["answers"][0]["answer_start"]
                
                data.append({
                    "context": context, 
                    "question": question, 
                    "answer": answer_text, 
                    "answer_start": answer_start, 
                    "answer_end": answer_start + len(answer_text)})


    data.sort(key=lambda x: (len(x["answer"]), len(x["question"])))
    # Limit the number of samples if specified
    if max_samples > 0:
        data = data[:max_samples]
    return data

def prepare_dataloaders_tokenizer():
    """
    Prepares the data loaders for training and validation sets.
    """
    tokenizer = BPETokenizer()
    train_data = load_and_process_squad("data/m2_train.json", max_samples=20000)
    dev_data = load_and_process_squad("data/m2_dev.json", max_samples=2000)

    context_max_length = 318
    question_max_length = 23
    answer_max_length = 6
    train_dataset = QADataset(train_data, tokenizer, context_max_length=context_max_length, question_max_length=question_max_length, answer_max_length=answer_max_length, include_context=True)
    dev_dataset = QADataset(dev_data, tokenizer, context_max_length=context_max_length, question_max_length=question_max_length, answer_max_length=answer_max_length, include_context=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # add this `num_workers=0` if you want to see print in __getitem__ in dataset class
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    return train_dataloader, dev_dataloader, tokenizer

def prepare_dataloaders_tokenizer_glove():
    """
    Prepares the data loaders for training and validation sets with GloVe embeddings.
    """
    tokenizer = GloveTokenizer()
    train_data = load_and_process_squad("data/m2_train.json", max_samples=20000)
    dev_data = load_and_process_squad("data/m2_dev.json", max_samples=2000)

    context_max_length = 249
    question_max_length = 20
    answer_max_length = 5
    train_dataset = QADatasetGlove(train_data, tokenizer, context_max_length=context_max_length, question_max_length=question_max_length, answer_max_length=answer_max_length, include_context=True, encode_two_texts=True)
    dev_dataset = QADatasetGlove(dev_data, tokenizer, context_max_length=context_max_length, question_max_length=question_max_length, answer_max_length=answer_max_length, include_context=True, encode_two_texts=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True) # add this `num_workers=0` if you want to see print in __getitem__ in dataset class
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
    return train_dataloader, dev_dataloader, tokenizer
