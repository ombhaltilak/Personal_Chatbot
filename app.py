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





# import sys
# for line in sys.stdin:
#     try:
#         user, action, time = line.strip().split(",")
#         print(f"{user}\t{action},{time}")
#     except:
#         continue



# import sys
# from datetime import datetime

# durations = {}
# logins = {}

# for line in sys.stdin:
#     user, val = line.strip().split("\t")
#     action, time_str = val.split(",")
#     t = datetime.strptime(time_str, "%H:%M")

#     if action == "login":
#         logins[user] = t
#     elif action == "logout" and user in logins:
#         duration = (t - logins[user]).total_seconds() / 60
#         durations[user] = durations.get(user, 0) + duration
#         del logins[user]
# if durations:
#     max_time = max(durations.values())
#     for u, d in durations.items():
#         if d == max_time:
#             print(f"{u}\t{d}")







# Predicting Admission Chances using 3 Separate Models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---- Sample dataset ----
data = {
    'GRE_Score': [320, 310, 305, 325, 330, 300, 290, 340, 315, 310],
    'GPA': [9.1, 8.5, 8.3, 9.5, 9.7, 8.0, 7.5, 9.9, 8.9, 8.7],
    'Research': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    'Chance_of_Admit': [0.92, 0.75, 0.68, 0.95, 0.97, 0.72, 0.65, 0.99, 0.88, 0.79]
}
df = pd.DataFrame(data)

# ---- Split dataset ----
X = df[['GRE_Score', 'GPA', 'Research']]
y = df['Chance_of_Admit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =====================================================
# 1️⃣ Linear Regression Model
# =====================================================
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)
lin_r2 = r2_score(y_test, lin_pred)
lin_mae = mean_absolute_error(y_test, lin_pred)

# =====================================================
# 2️⃣ Decision Tree Model
# =====================================================
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
tree_r2 = r2_score(y_test, tree_pred)
tree_mae = mean_absolute_error(y_test, tree_pred)

# =====================================================
# 3️⃣ Random Forest Model
# =====================================================
rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

# =====================================================
# 📊 Combine Results into a Table
# =====================================================
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'R2_Score': [round(lin_r2, 3), round(tree_r2, 3), round(rf_r2, 3)],
    'MAE': [round(lin_mae, 3), round(tree_mae, 3), round(rf_mae, 3)]
})

print("📊 Model Evaluation Results:")
print(results)
