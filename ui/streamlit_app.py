import streamlit as st
import requests

# API Configuration (Ensure This Matches FastAPI's Port)
API_URL = "http://127.0.0.1:8003"

st.title("📄 AI-Powered RAG with NVIDIA NIM, ChromaDB & LangChain Agent")

# **Session State for Chat History**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# **PDF Upload Section**
st.header("📤 Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Send file to FastAPI backend
    with open(temp_file_path, "rb") as f:
        files = {"file": (uploaded_file.name, f, "application/pdf")}
        response = requests.post(f"{API_URL}/upload/", files=files)

    # Handle response
    if response.status_code == 200:
        st.success(response.json()["message"])
    else:
        st.error(f"❌ Failed to upload PDF. Error: {response.text}")

    os.remove(temp_file_path)

# **Chat Interface**
st.header("💬 Chat with AI")

query = st.text_input("Enter your question:")
if st.button("Submit Query"):
    if query:
        # Construct payload with chat history
        payload = {
            "question": query,
            "chat_history": st.session_state.chat_history  # Maintain conversation context
        }
        response = requests.post(f"{API_URL}/query/", json=payload)

        if response.status_code == 200:
            response_data = response.json()
            # Append conversation to chat history
            st.session_state.chat_history.append({"content": query, "type": "human"})
            st.session_state.chat_history.append({"content": response_data['answer'], "type": "ai"})

            # Display chat
            st.write("### 📝 Conversation History")
            for chat in st.session_state.chat_history:
                role = "🧑‍💻 You: " if chat["type"] == "human" else "🤖 AI: "
                st.write(f"{role}{chat['content']}")

        else:
            st.error(f"❌ Failed to process query: {response.status_code} - {response.text}")
    else:
        st.warning("⚠️ Please enter a question before submitting.")
