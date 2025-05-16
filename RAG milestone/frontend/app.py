import gradio as gr
import requests
from typing import List, Tuple

def respond(user_message: str, history: List[Tuple[str, str]]):
    """
    Responds via a streaming generator.
    """
    if not user_message:
        yield "", history, history
        return

    url = "http://localhost:8000/ask"
    payload = {"question": user_message}

    # Open the streaming connection
    with requests.post(url, json=payload, stream=True) as resp:
        answer = ""
        history = history + [(user_message, answer)]
        
        # Yield the empty response immediately to trigger UI update
        yield "", history, history  

        # Read and process each streamed chunk
        for chunk in resp.iter_content(chunk_size=1, decode_unicode=True):
            if chunk:
                answer += chunk
                history[-1] = (user_message, answer)
                yield "", history, history  # Continuously update UI

with gr.Blocks() as demo:
    gr.Markdown("## Chat with the RAG System")
    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(placeholder="Enter your message", show_label=False)
        send_btn = gr.Button("Send")

    txt.submit(respond, [txt, state], [txt, chatbot, state])
    send_btn.click(respond, [txt, state], [txt, chatbot, state])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)