from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast, pipeline
import os 
import numpy as np
import torch # Add this import

class DistilBert:
    '''
    DistilBERT model for question answering.
    '''
    def __init__(self, 
                 tokenizer_path: str = "./distilbert-squad-finetuned_tokenizer", 
                 model_path: str = "./distilbert-squad-finetuned_model",
                 use_memory: bool = False):
        '''
        Initializes the DistilBERT model and tokenizer.

        Args
        '''
        self.device = "cpu"
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_path).to(self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
        self.qa = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1
        )
        self.use_memory = use_memory
        self.memory = []

    def invoke(self, question: str, contexts: str) -> str:
        '''
        Generates a response from the model.

        Args:
        - user_message (str): The user message

        Returns:
        - str: The response from the model
        '''
        if self.use_memory:
            context = self.get_memory_string()
            contexts = context + "\n" + contexts
        answer =  self.answer_question(question=question, context=contexts)
        if self.use_memory:
            self.memory.append({"human": question.strip(), "ai": answer.strip()})
        return answer.strip()
    
    def get_memory_string(self) -> str:
        '''
        Formats memory into a string for the model.

        Returns:
        - str: The memory string
        '''
        return "\n".join([f"( Human: {item['human']}, AI: {item['ai']} )" for item in self.memory])
    


    def preprocess_for_dev(
        self,
        question: str,
        context: str,
        max_length: int = 384,
        stride: int = 128,
    ):
        """
        Tokenize a single question/context into overlapping chunks for inference.
        Returns:
        - inputs: dict of tensors you can pass directly to model(**inputs)
        - sample_map: for each chunk index i, sample_map[i] == original example index (always 0 here)
        - offset_mappings: a list of offset lists (len = #chunks) so you can map token positions back to char spans.
        """
        # strip & tokenize
        inputs = self.tokenizer(
            question.strip(),
            context,
            truncation="only_second",
            padding="max_length",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        # grab the spill-over mapping and raw offsets
        sample_map = inputs.pop("overflow_to_sample_mapping").tolist()
        offset_mappings = inputs.pop("offset_mapping").tolist()
        # mask out offsets for question tokens (so only context spans remain)
        for i, mapping in enumerate(offset_mappings):
            seq_ids = inputs.sequence_ids(i)
            offset_mappings[i] = [
                off if seq_ids[k] == 1 else None
                for k, off in enumerate(mapping)
            ]
        return inputs, sample_map, offset_mappings


    def answer_question(
        self,
        question: str,
        context: str,
        max_length: int = 384,
        stride: int = 128,
    ) -> str:
        """
        Given a question & context, run the model and pick the highest-scoring span.
        """
        # inputs, sample_map, offset_mappings = self.preprocess_for_dev(
        #     question, context, max_length, stride
        # )
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # with torch.no_grad():
        #     outputs = self.model(**inputs) 
        # start_logits = outputs.start_logits.cpu().numpy()
        # end_logits   = outputs.end_logits.cpu().numpy()

        # # 3) for each chunk, find best span
        # best_answer = {"text": "", "score": -float("inf")}
        # # Ensure that inputs['input_ids'] is on the same device as the model if you move inputs to a device
        # # For example, if your model is on GPU: inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # num_chunks = inputs["input_ids"].shape[0]

        # for chunk_idx in range(num_chunks):
        #     # Get logits for the current chunk
        #     current_start_logits = start_logits[chunk_idx]
        #     current_end_logits = end_logits[chunk_idx]
        #     current_offset_mapping = offset_mappings[chunk_idx]
            
        #     start_indexes = np.argsort(current_start_logits)[::-1]
        #     end_indexes = np.argsort(current_end_logits)[::-1]

        #     valid_answers = []
        #     for start_index in start_indexes:
        #         for end_index in end_indexes:
        #             # Consider answers that are not in the question part and are valid
        #             if (
        #                 current_offset_mapping[start_index] is None
        #                 or current_offset_mapping[end_index] is None
        #             ):
        #                 continue
        #             if end_index < start_index:
        #                 continue
                    
        #             score = current_start_logits[start_index] + current_end_logits[end_index]
        #             text = context[current_offset_mapping[start_index][0] : current_offset_mapping[end_index][1]]
        #             valid_answers.append({"score": score, "text": text})

        #     if valid_answers:
        #         chunk_best_answer = max(valid_answers, key=lambda x: x["score"])
        #         if chunk_best_answer["score"] > best_answer["score"]:
        #             best_answer = chunk_best_answer
                    
        # return best_answer["text"] 
        result = self.qa(
            question=question,
            context=context,
            max_length=max_length,
            stride=stride,
            handle_impossible_answer=True,
            topk=1,
        )
        # pipeline returns a dict or a list of dicts if topk>1
        if isinstance(result, list):
            return result[0]["answer"]
        return result["answer"]
    
if __name__ == "__main__":
    print('-' * 50)
    model_instance = DistilBert(use_memory=True) # Renamed to model_instance to avoid confusion with self.model
    question = "What is the capital of France?"
    context = "Paris is the capital of France. It is a beautiful city."
    result_invoke = model_instance.invoke(question, context)
    print(f"Answer from invoke: {result_invoke}")
    print('-' * 50)
    question2 = "What is the capital of Germany?"
    context2 = "Berlin is the capital of Germany. It is known for its history."
    result_invoke2 = model_instance.invoke(question2, context2)
    print(f"Answer from invoke: {result_invoke2}")
    print("Memory after two questions:")
    print(model_instance.get_memory_string())

    print('-' * 50)
    question3 = "What was my full last question?"
    context3 = "This is a random context."
    result_invoke2 = model_instance.invoke(question3, context3)
    print(f"Answer from invoke: {result_invoke2}")
