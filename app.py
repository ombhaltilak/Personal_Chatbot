import streamlit as st
import os
import requests
import tempfile
from llama_index.llms.llama_cpp import LlamaCPP   # ✅ fixed import
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ====== CHANGE THIS TO YOUR PUBLIC MODEL URL ======
MODEL_URL = "https://huggingface.co/your-username/your-model/resolve/main/llama-2-7b-chat.Q2_K.gguf"

# --------------------------------------------------

def download_model(url):
    """Download the model to a temp directory if not already present."""
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(url))
    if not os.path.exists(local_path):
        with st.spinner("Downloading model... (this may take a while)"):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return local_path

def init_page():
    st.set_page_config(page_title="Personal Chatbot")
    st.header("Personal Chatbot")
    st.sidebar.title("Options")

def select_llm():
    model_path = download_model(MODEL_URL)
    return LlamaCPP(
        model_path=model_path,
        temperature=1.9,
        max_new_tokens=150,
        context_window=4096,
        model_kwargs={
            "n_threads": max(1, os.cpu_count() - 2),
            "n_batch": 8,
            "n_gpu_layers": 20,
            "verbose": False
        },
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False,
    )

def init_messages():
    if st.sidebar.button("Clear Conversation", key="clear") or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant. Reply in markdown format.")
        ]

def get_answer(llm, messages):
    prompt = ""
    for msg in messages:
        role = "System" if isinstance(msg, SystemMessage) else \
               "User" if isinstance(msg, HumanMessage) else \
               "Assistant"
        prompt += f"{role}: {msg.content}\n"

    response = llm.complete(prompt)
    return response.text

def main():
    init_page()
    llm = select_llm()
    init_messages()

    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Bot is typing..."):
            answer = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))

    for message in st.session_state.get("messages", []):
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

if __name__ == "__main__":
    main()

