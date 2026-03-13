import streamlit as st
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredExcelLoader
import tempfile


st.set_page_config(page_title="Chatbot prototype for prediabeat", layout="wide")

# Model LLM Hugging Face
HF_MODEL_ID = "Llama-3.1-8B-Instruct?inference_provider=novita" 
# Model embedding 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG ENGINE
device = "cuda" if torch.cuda.is_available() else "cpu"
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        device_map=device,
        dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# def process_combined_knowledge(uploaded_files, manual_text):
#     all_docs = []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
#     # 1. Proses dari PDF/TXT
#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_path = tmp_file.name
            
#             loader = PyPDFLoader(tmp_path) if uploaded_file.name.endswith('.pdf') else TextLoader(tmp_path)
#             docs = loader.load()
#             all_docs.extend(text_splitter.split_documents(docs))
#             os.remove(tmp_path)

#     # 2. Proses dari Manual Text (Tambahkan ke list yang sama)
#     if manual_text:
#         # Ubah teks manual menjadi objek Document agar seragam
#         from langchain_core.documents import Document
#         manual_docs = [Document(page_content=manual_text, metadata={"source": "manual"})]
#         all_docs.extend(text_splitter.split_documents(manual_docs))

#     if not all_docs:
#         return None

#     # Simpan semua ke satu Vector Store
#     return FAISS.from_documents(all_docs, get_embeddings())

#experiment
def process_combined_knowledge(uploaded_files, manual_text):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Dapatkan ekstensi file asli
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Gunakan penamaan manual untuk sementara agar tidak konflik akses file
            tmp_path = f"temp_{uploaded_file.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                if file_extension == '.pdf':
                    loader = PyPDFLoader(tmp_path)
                elif file_extension == '.txt':
                    # SOLUSI: Tambahkan encoding='utf-8' untuk TextLoader
                    loader = TextLoader(tmp_path, encoding='utf-8')
                elif file_extension == '.csv':
                    # SOLUSI: Tambahkan encoding='utf-8' untuk CSVLoader
                    loader = CSVLoader(tmp_path, encoding='utf-8')
                elif file_extension in ['.xlsx', '.xls']:
                    loader = UnstructuredExcelLoader(tmp_path)
                else:
                    st.warning(f"Format {file_extension} tidak didukung.")
                    continue
                
                docs = loader.load()
                all_docs.extend(text_splitter.split_documents(docs))
            
            except Exception as e:
                st.error(f"Gagal memproses {uploaded_file.name}: {e}")
            
            finally:
                # Pastikan file dihapus setelah diproses
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # 2. Proses dari Manual Text (DI-COMMENT SEPERTI PERMINTAAN)
    # if manual_text:
    #     from langchain_core.documents import Document
    #     manual_docs = [Document(page_content=manual_text, metadata={"source": "manual"})]
    #     all_docs.extend(text_splitter.split_documents(manual_docs))

    if not all_docs:
        return None

    return FAISS.from_documents(all_docs, get_embeddings())

def process_documents(text_input):
    """Memecah teks domain menjadi bagian kecil dan menyimpannya ke Vector Store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(text_input)
    vector_db = FAISS.from_texts(docs, get_embeddings())
    return vector_db

# ========================= SESSION STATE =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# ========================= UI LAYOUT =========================
st.title("Chatbot Prediabeat")

# --- SIDEBAR: KNOWLEDGE BASE ---
with st.sidebar:
    st.header("📁 Knowledge Base")
    
    # Input via File Upload (Bisa banyak file sekaligus)
    uploaded_files = st.file_uploader(
        "Upload Dokumen Domain (PDF/TXT/CSV/XLSX)", 
        type=["pdf", "txt", "csv", "xlsx"], 
        accept_multiple_files=True
    )
    
    # Input via Text Area (Opsional, tetap dipertahankan jika butuh input manual)
    st.markdown("--- atau masukkan teks manual ---")
    manual_text = st.text_area("Input Teks Manual:", height=150)
    
    if st.button("Update Knowledge Base"):
        with st.spinner("Menggabungkan data..."):
        # Panggil fungsi baru yang menggabungkan keduanya
            st.session_state.vector_db = process_combined_knowledge(uploaded_files, manual_text)
            st.success("Knowledge Base diperbarui (PDF + Manual)!")

# --- MAIN CHAT INTERFACE ---
# Tampilkan chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Chat
if prompt := st.chat_input("Tanyakan sesuatu tentang domain..."):
    # Tambahkan pesan user ke history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Logika RAG
    if st.session_state.vector_db is None:
        with st.chat_message("assistant"):
            st.warning("Mohon isi Knowledge Base di sidebar terlebih dahulu.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Mencari referensi..."):
                # 1. Cari teks relevan dari database (Retrieval)
                search_results = st.session_state.vector_db.similarity_search(prompt, k=5)
                context = "\n".join([doc.page_content for doc in search_results])

                # 2. Strict Prompting (Instruction)
                system_prompt = (
                    "Anda adalah asisten ahli yang hanya boleh menjawab berdasarkan KONTEKS yang diberikan."
                    "Jika jawaban tidak ada di dalam KONTEKS, katakan dengan sopan bahwa Anda tidak tahu "
                    "atau informasi tersebut di luar domain Anda. JANGAN gunakan pengetahuan luar.\n\n"
                    f"KONTEKS:\n{context}"
                )

                # 3. Generate Jawaban
                pipe = load_llm()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                # Format prompt untuk Qwen
                formatted_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outputs = pipe(formatted_prompt, do_sample=False, temperature=0.1)
                full_response = outputs[0]["generated_text"].split("assistant")[-1].strip()
                
                # Bersihkan sisa token khusus
                clean_response = full_response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

                st.markdown(clean_response)
                st.session_state.messages.append({"role": "assistant", "content": clean_response})

# Tombol Reset
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()