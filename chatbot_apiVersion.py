import streamlit as st
from huggingface_hub import InferenceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader

st.set_page_config(page_title="Chatbot Prediabeat (API Version)", layout="wide")

# ========================= KONFIGURASI API =========================
# Ganti dengan model Sahabat-AI atau Qwen sesuai keinginanmu
HF_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct" 
HF_TOKEN = st.secrets["huggingface_token"] # Masukkan token Hugging Face Anda di sini

# Model embedding (tetap lokal karena ringan dan cepat)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def get_inference_client():
    return InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ========================= RAG ENGINE =========================
def process_combined_knowledge(uploaded_files, manual_text):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    
    # 1. Proses File Upload
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            tmp_path = f"temp_{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                if file_extension == '.pdf': loader = PyPDFLoader(tmp_path)
                elif file_extension == '.txt': loader = TextLoader(tmp_path, encoding='utf-8')
                elif file_extension == '.csv': loader = CSVLoader(tmp_path, encoding='utf-8')
                else: continue
                
                all_docs.extend(text_splitter.split_documents(loader.load()))
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

    # 2. Proses Manual Text
    if manual_text.strip():
        all_docs.extend(text_splitter.create_documents([manual_text]))

    if not all_docs: return None
    return FAISS.from_documents(all_docs, get_embeddings())

# ========================= SESSION STATE & UI =========================
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None

st.title("Prediabeat Assistant")

with st.sidebar:
    st.header("📁 Knowledge Base")
    uploaded_files = st.file_uploader("Upload Dokumen", type=["pdf", "txt", "csv"], accept_multiple_files=True)
    manual_text = st.text_area("Input Teks Manual:", height=150)
    
    if st.button("Update Knowledge Base"):
        with st.spinner("Memproses data..."):
            st.session_state.vector_db = process_combined_knowledge(uploaded_files, manual_text)
            st.success("Knowledge Base diperbarui!")

# Tampilkan Chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# Input Chat
if prompt := st.chat_input("Tanyakan sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if st.session_state.vector_db is None:
        st.warning("Mohon isi Knowledge Base terlebih dahulu.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                # 1. Retrieval
                search_results = st.session_state.vector_db.similarity_search(prompt, k=3)
                context = "\n".join([doc.page_content for doc in search_results])

                # 2. System Instruction
                system_instr = (
                    # "Anda adalah Prediabeat Assistant. Gunakan KONTEKS berikut untuk menjawab. "
                    # "Jika tidak ada di konteks, katakan maaf dan tawarkan bantuan lain. "
                    # "Bukan dokter, dilarang memberi dosis obat.\n\n"
                    # f"KONTEKS:\n{context}"
                     "Anda adalah asisten ahli yang hanya boleh menjawab berdasarkan KONTEKS yang diberikan."
                    "Jika jawaban tidak ada di dalam KONTEKS, katakan dengan sopan bahwa Anda tidak tahu "
                    "atau informasi tersebut di luar domain Anda. JANGAN gunakan pengetahuan luar.\n\n"
                    f"KONTEKS:\n{context}"
                )

                # 3. API Call
                client = get_inference_client()
                response = ""
                
                # Menggunakan stream agar UI lebih interaktif
                placeholder = st.empty()
                messages_for_api = [
                    {"role": "system", "content": system_instr},
                    {"role": "user", "content": prompt}
                ]

                for message in client.chat_completion(
                    messages=messages_for_api,
                    max_tokens=800,
                    stream=True,
                    temperature=0.3 # Suhu rendah agar lebih faktual sesuai RAG
                ):
                    token = message.choices[0].delta.content
                    if token:
                        response += token
                        placeholder.markdown(response + "▌")
                
                placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()