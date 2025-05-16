import os
from logger import get_logger  # Import the custom logger
from datetime import datetime

class ContextLogger:
    def __init__(self, log_dir="logs"):
        # Ensure the directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.log_dir = log_dir
        self.logger = get_logger()  # Get the custom logger

    def log_retrieved_documents(self, user_message, retrieved_documents):
        # Create a timestamp for the log file
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create a log file with the timestamped filename
        log_filename = os.path.join(self.log_dir, f"{timestamp}.txt")
        
        # Write the question (user_message) and the documents split into paragraphs
        with open(log_filename, "w") as file:
            file.write(f"Question: {user_message}\n\n")
            for i, doc in enumerate(retrieved_documents):
                paragraphs = doc.page_content.split('\n')  # Split the content into paragraphs
                file.write(f"Document {i+1}:\n")
                for paragraph in paragraphs:
                    file.write(f"{paragraph}\n\n")
                file.write("\n---\n\n")
        
        self.logger.debug(f"Logged retrieved documents to {log_filename}")