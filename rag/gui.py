import gradio as gr
import threading
from queue import Queue

from rag import get_answer

def user(user_message, history):
    return "", history + [{"role": "user", "content": user_message}]

def bot(history):
    query = history[-1]["content"]

    token_queue = Queue()

    def run_get_answer():
        get_answer(query, token_queue=token_queue)
        token_queue.put(None)

    threading.Thread(target=run_get_answer, daemon=True).start()

    bot_message = ""
    history.append({"role": "assistant", "content": ""})

    while True:
        token = token_queue.get()
        if token is None:
            break
        bot_message += token
        history[-1]["content"] = bot_message
        yield history

with gr.Blocks() as demo:
    gr.Markdown("# HSE RAG chatbot")

    chatbot = gr.Chatbot(height=500, type='messages')

    msg = gr.Textbox(
        placeholder='Введите сообщение',
        show_label=False,
    )

    clear = gr.Button("Очистить историю сообщений")

    msg.submit(
        user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        bot,
        inputs=chatbot,
        outputs=chatbot
    )

    clear.click(
        fn=lambda: [],
        inputs=None,
        outputs=chatbot,
        queue=False
    )

demo.launch(share=True)
