import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.tools.tavily_search import TavilySearchResults
from PIL import Image
import pytesseract
import tempfile
import os
import speech_recognition as sr
import io

# -------------------------
# Streamlit UI Setup
# -------------------------
st.set_page_config(page_title="Multi-Modal RAG Chatbot", layout="wide")
st.title("🧠 Multi-Modal RAG Chatbot with Tavily Search, OCR, and Speech Input")

# -------------------------
# Sidebar: File & Audio Upload
# -------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, Word, or Image files", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True
)
audio_file = st.sidebar.file_uploader("Upload an audio query (optional)", type=["wav", "mp3", "m4a"])

# -------------------------
# Initialize Embedding Model & Vector DB
# -------------------------
embedding_model = OpenAIEmbeddings()
vector_db = None

temp_dir = tempfile.mkdtemp()
all_docs = []

# -------------------------
# OCR function for extracting text from images
# -------------------------
def extract_text_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return text

# -------------------------
# Load and Chunk Documents
# -------------------------
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        elif file.name.endswith((".png", ".jpg", ".jpeg")):
            text = extract_text_from_image(file.getvalue())
            documents = [{"page_content": text, "metadata": {"source": file.name}}]
        else:
            continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(documents)
        all_docs.extend(docs)

    vector_db = Chroma.from_documents(documents=all_docs, embedding=embedding_model, persist_directory="./chroma_store")

# -------------------------
# Audio Input Transcription
# -------------------------
user_query = None
if audio_file:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            user_query = recognizer.recognize_google(audio_data)
            st.sidebar.success(f"Transcribed Query: {user_query}")
        except sr.UnknownValueError:
            st.sidebar.error("Could not understand audio.")

# -------------------------
# Setup Conversational RAG Chain
# -------------------------
if vector_db:
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, verbose=True
    )

    tavily_search = TavilySearchResults(k=3)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    text_input = st.chat_input("Ask me anything about your documents or search externally...")
    if text_input:
        user_query = text_input

    if user_query:
        with st.spinner("Thinking..."):
            response = rag_chain.run(user_query)
            if not response or len(response.strip()) < 50:
                tavily_results = tavily_search.run(user_query)
                response = f"(🔍 External Info via Tavily)\n\n{tavily_results}"
            st.session_state.chat_history.append((user_query, response))

    for query, resp in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(query)
        with st.chat_message("assistant"):
            st.write(resp)

else:
    st.info("Please upload at least one document or image to begin contextual chat.")
