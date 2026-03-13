import streamlit as st
import re
import tempfile
import os
import subprocess
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# ========================= CONFIG =========================
st.set_page_config(page_title="AI Interview Assessment", layout="wide")

HF_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
HF_WHISPER_MODEL = "openai/whisper-medium"

DEFAULT_QUESTIONS = [
    "Can you share any specific challenges you faced while working on certification and how you overcame them?",
    "Can you describe your experience with transfer learning in TensorFlow? How did it benefit your projects?",
    "Describe a complex TensorFlow model you have built and the steps you took to ensure its accuracy and efficiency.",
    "Explain how to implement dropout in a TensorFlow model and the effect it has on training.",
    "Describe the process of building a convolutional neural network (CNN) using TensorFlow for image classification."
]

FFMPEG_EXE_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe" # Pastikan path benar
CHUNK_LENGTH_S = 30
STRIDE_LENGTH_S = 5

# ========================= PIPELINE (CACHED) =========================
@st.cache_resource
def get_asr_pipeline():
    return pipeline(
        task="automatic-speech-recognition",
        model=HF_WHISPER_MODEL,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S
    )

@st.cache_resource
def get_llm_pipeline():
    print(f"🔄 Memuat model {HF_MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            HF_MODEL_ID,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation="eager" 
        )
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500
        )
        print("✅ Model LLM berhasil dimuat!")
        return pipe
    except Exception as e:
        print(f"❌ Error Load Model: {e}")
        return None

# ========================= CORE FUNCTIONS =========================
def transcribe_video(video_bytes):
    asr = get_asr_pipeline()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        subprocess.run(
            [FFMPEG_EXE_PATH, "-y", "-i", tmp_in_path, "-ac", "1", "-ar", "16000", tmp_out_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        result = asr(tmp_out_path)
        text = result.get("text", "") if isinstance(result, dict) else result[0].get("text", "")
        return text
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        try: os.remove(tmp_in_path)
        except: pass
        try: os.remove(tmp_out_path)
        except: pass

def analyze_answer(question, answer):
    pipe = get_llm_pipeline()
    if pipe is None: return "ERROR: Model not loaded.", 0

    system_prompt = (
        "You are an expert technical interviewer. Evaluate the candidate's answer based ONLY on the provided transcript. "
        "Specific technical keywords should result in higher scores.\n"
        "Scoring:\n0=No Answer, 1=Irrelevant, 2=General, 3=Good/Practical, 4=Expert/Deep.\n"
        "Output Format MUST be exactly:\n"
        "SKOR: <0-4>\n"
        "ALASAN: <Short explanation>"
    )
    user_content = f"Question: {question}\n\nCandidate Answer: {answer}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

    try:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=300, do_sample=False, temperature=0.1)
        generated = outputs[0]["generated_text"]
        
        response = generated.split("assistant")[-1].strip()
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        
        score_match = re.search(r"KLASIFIKASI[:\- ]*([0-4])", response)
        score = int(score_match.group(1)) if score_match else 0
        
        return response, score
    except Exception as e:
        return f"Error: {e}", 0

# ========================= SESSION STATE INIT =========================
# Kita gunakan List of Dictionaries untuk menampung data dinamis
if "interview_items" not in st.session_state:
    # Init awal: Masukkan 5 Pertanyaan Default
    st.session_state.interview_items = [
        {"question": q, "transcript": "", "response": "", "score": 0, "processed": False} 
        for q in DEFAULT_QUESTIONS
    ]

# ========================= UI LAYOUT =========================
st.title("Candidate Speech Analyzer")

# --- HEADER & CONTROLS ---
col_name, col_actions = st.columns([2, 2])

with col_name:
    nama_pelamar = st.text_input("Nama Pelamar", placeholder="Contoh: Budi Santoso")

with col_actions:
    st.write("**Pengaturan Soal**")
    c1, c2, c3 = st.columns(3)
    
    # Tombol 1: Load Default 5
    if c1.button("Reset to Default 5"):
        st.session_state.interview_items = [
            {"question": q, "transcript": "", "response": "", "score": 0, "processed": False} 
            for q in DEFAULT_QUESTIONS
        ]
        st.rerun()

    # Tombol 2: Tambah Pertanyaan Baru (Iterasi)
    if c2.button("Tambah Penilaian"):
        st.session_state.interview_items.append(
            {"question": "", "transcript": "", "response": "", "score": 0, "processed": False}
        )
        st.rerun()
        
    # Tombol 3: Hapus Semua
    if c3.button("Hapus Semua"):
        st.session_state.interview_items = []
        st.rerun()

st.markdown("---")

# --- DYNAMIC LIST LOOP ---
# Kita loop berdasarkan jumlah item yang ada di session_state
total_score_accumulated = 0
processed_count = 0

if not st.session_state.interview_items:
    st.info("Belum ada pertanyaan. Silakan klik 'Tambah Soal' atau 'Reset to Default'.")

for idx, item in enumerate(st.session_state.interview_items):
    with st.container():
        c_q, c_vid, c_act = st.columns([3, 2, 1])
        
        
        with c_q:
            # Pertanyaan bisa diedit user
            new_q = st.text_area(
                f"Pertanyaan {idx+1}", 
                value=item['question'], 
                key=f"q_{idx}", 
                height=70
            )
            # Update state jika user mengetik
            st.session_state.interview_items[idx]['question'] = new_q

        with c_vid:
            vid = st.file_uploader(f"Video {idx+1}", type=["mp4", "mov", "wav"], key=f"v_{idx}", label_visibility="collapsed")
            if vid:
                st.caption(f"📁{vid.name}")

        with c_act:
            st.write("##")
            # Tombol Analisis Per Item
            if st.button("Nilai", key=f"btn_{idx}", disabled=not vid):
                if not new_q:
                    st.warning("Isi pertanyaan dulu!")
                else:
                    with st.spinner("Menganalisis..."):
                        # 1. Transkrip
                        trans = transcribe_video(vid.read())
                        # 2. AI Scoring
                        resp, sc = analyze_answer(new_q, trans)
                        
                        # 3. Update State
                        st.session_state.interview_items[idx]['transcript'] = trans
                        st.session_state.interview_items[idx]['response'] = resp
                        st.session_state.interview_items[idx]['score'] = sc
                        st.session_state.interview_items[idx]['processed'] = True
                        st.rerun() # Refresh agar hasil muncul

    # Tampilkan Hasil di bawah item tersebut jika sudah diproses
    if item['processed']:
        processed_count += 1
        total_score_accumulated += item['score']
        
        with st.expander(f"Hasil Penilaian Soal {idx+1} (Skor: {item['score']})", expanded=True):
            st.markdown(f"**Transkrip:**\n> {item['transcript']}")
            st.markdown(f"**Alasan AI:**")
            st.code(item['response'], language="yaml")
    
    st.divider()

# --- SIDEBAR REKAP ---
st.sidebar.title("Rekapitulasi")
st.sidebar.write(f"**Pelamar:** {nama_pelamar if nama_pelamar else '-'}")

# Hitung Rata-rata Dinamis
# Pembagi adalah jumlah video yang SUDAH dinilai (processed_count), bukan fixed 5
if processed_count > 0:
    final_avg = total_score_accumulated / processed_count
    
    st.sidebar.metric("Item Dinilai", f"{processed_count} / {len(st.session_state.interview_items)}")
    st.sidebar.metric("Total Poin", total_score_accumulated)
    st.sidebar.metric("Rata-rata Skor", f"{final_avg:.2f} / 4.0")
    
    st.sidebar.markdown("---")
    
    if final_avg >= 3:
        st.sidebar.success("REKOMENDASI: **LOLOS**")
    elif final_avg >= 2:
        st.sidebar.warning("REKOMENDASI: **PERTIMBANGKAN**")
    else:
        st.sidebar.error("REKOMENDASI: **TIDAK LOLOS**")
else:
    st.sidebar.info("Silakan upload video dan klik 'Nilai' untuk melihat hasil.")