import streamlit as st
import re
import tempfile
import os
from transformers import pipeline
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========================= CONFIG =========================
st.set_page_config(page_title="AI Interview Assessment", layout="wide")

HF_PHI3_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HF_WHISPER_MODEL = "openai/whisper-medium"

INTERVIEW_QUESTIONS = [
    "Can you share any specific challenges you faced while working on certification and how you overcame them?",
    "Can you describe your experience with transfer learning in TensorFlow? How did it benefit your projects?",
    "Describe a complex TensorFlow model you have built and the steps you took to ensure its accuracy and efficiency.",
    "Explain how to implement dropout in a TensorFlow model and the effect it has on training.",
    "Describe the process of building a convolutional neural network (CNN) using TensorFlow for image classification."
]

CRITERIA = (
    c
)

FFMPEG_EXE_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CHUNK_LENGTH_S = 30  # Panjang chunk dalam detik (misalnya, 30 detik)
STRIDE_LENGTH_S = 5  # Panjang overlap (stride) dalam detik (misalnya, 5 detik)
# ========================= PIPELINE CACHE =========================
@st.cache_resource
def get_asr_pipeline():
    return pipeline(
        task="automatic-speech-recognition",
        model=HF_WHISPER_MODEL,
        device_map = "auto",
        chunk_length_s = CHUNK_LENGTH_S,
        stride_length_s = STRIDE_LENGTH_S
    )


@st.cache_resource
def get_llm_pipeline():
    print("🔄 Sedang memuat model Phi-3 dengan konfigurasi MANUAL...")
    try:
        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            HF_PHI3_MODEL, 
            trust_remote_code=True
        )
        
        # 2. Load Model dengan 'attn_implementation="eager"' (PENTING!)
        model = AutoModelForCausalLM.from_pretrained(
            HF_PHI3_MODEL,
            device_map="cuda" if torch.cuda.is_available() else "cpu", 
            torch_dtype="auto", 
            trust_remote_code=True,
            attn_implementation="eager"  # <--- INI SOLUSI ERROR DynamicCache
        )

        # 3. Buat Pipeline dari model yang sudah diload
        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500  # Panjang output maksimal
        )
        
        print("✅ Model Phi-3 berhasil dimuat!")
        return pipe
        
    except Exception as e:
        print(f"❌ GAGAL MEMUAT MODEL: {e}")
        # Kembalikan None agar error bisa ditangkap di phi3_api
        return None

# ========================= FUNCTIONS =========================
def transcribe_via_hf(video_bytes):
    """
    Transkripsi video/audio menggunakan Whisper lokal di HF Space.
    """
    asr = get_asr_pipeline()

    # Simpan input video sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in.flush()
        tmp_in_path = tmp_in.name

    # Simpan output audio sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
        tmp_out_path = tmp_out.name

    # Convert to WAV mono 16k
    try:
        subprocess.run(
            [
                FFMPEG_EXE_PATH, "-y", "-i", tmp_in_path,
                "-ac", "1", "-ar", "16000",
                tmp_out_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except Exception as e:
        return f"ERROR FFMPEG CONVERT: {e}"

    # Transcribe WAV
    try:
        result = asr(tmp_out_path)
        if isinstance(result, dict):
            return result.get("text", "")
        elif isinstance(result, list) and isinstance(result[0], dict):
            return result[0].get("text", "")
        return str(result)
    except Exception as e:
        return f"ERROR TRANSCRIBE: {e}"
    finally:
        try: os.remove(tmp_in_path)
        except: pass
def llm_api(question, answer):
    pipe = get_llm_pipeline()
    
    if pipe is None:
        return "ERROR: Model gagal dimuat. Cek terminal."

    # System Prompt: Aturan main
    system_prompt = (
        "You are an expert technical interviewer. Evaluate the candidate's answer based ONLY on the provided transcript. "
        "Specific technical keywords (like 'dropout', 'layers', 'validation loss') should result in higher scores.\n"
        "Output Format MUST be exactly:\n"
        "KLASIFIKASI: <0-4>\n"
        "ALASAN: <Explanation in English>"
    )

    # User Prompt: Data soal & jawaban
    user_content = f"Question: {question}\n\nCandidate Answer: {answer}"

    # Format Chat (PENTING untuk Qwen/Phi-3)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        # Generate prompt sesuai format model
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Eksekusi
        outputs = pipe(prompt, max_new_tokens=300, do_sample=False, temperature=0.1)
        
        # Bersihkan output
        generated_text = outputs[0]["generated_text"]
        # Ambil hanya bagian respons asisten (menghapus prompt)
        if "<|im_start|>assistant" in generated_text:
            return generated_text.split("<|im_start|>assistant")[-1].strip()
        elif "assistant\n" in generated_text: # Fallback untuk model lain
             return generated_text.split("assistant\n")[-1].strip()
            
        return generated_text
    except Exception as e:
        return f"ERROR INFERENCE: {e}"


# def phi3_api(prompt):
#     llm = get_llm_pipeline()

#     if llm is None:
#         return "ERROR: Model gagal dimuat. Cek terminal."

#     try:
#         # Phi-3 instruksi format: <|user|>\n ... <|end|>\n<|assistant|>
#         formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
#         out = llm(formatted_prompt, max_new_tokens=200, do_sample=False)
        
#         if isinstance(out, list) and len(out) > 0:
#             # Bersihkan output dari prompt input agar rapi
#             generated_text = out[0]["generated_text"]
#             if "<|assistant|>" in generated_text:
#                 return generated_text.split("<|assistant|>")[-1].strip()
#             return generated_text
#         return str(out)
#     except Exception as e:
#         return f"ERROR INFERENCE: {e}"


# def prompt_for_classification(question, answer):
#     return (
#         "You are an expert HR interviewer and technical evaluator. Your task is to objectively assess the "
#         "candidate's response based solely on the provided transcript. You must classify the answer using a strict "
#         "0 until 4 scoring rubric.\n\n"

#         f"{CRITERIA}\n\n"

#         "Evaluation Rules:\n"
#         "- Evaluate ONLY based on the candidate's answer.\n"
#         "- Do NOT add missing information, assumptions, or corrections.\n"
#         "- Judge relevance, accuracy, clarity, and depth based on the rubric.\n"
#         "- Your explanation must be concise and directly tied to the rubric.\n"
#         "- You MUST follow the output format exactly.\n\n"

#         "Required Output Format:\n"
#         "KLASIFIKASI: <angka>\n"
#         "ALASAN: <string>\n"
#     )

def parse_model_output(text):
    # 1. Pastikan teks bukan string error dari phi3_api
    if text.startswith("ERROR:"):
        return None, text # Kembalikan skor None dan error sebagai alasan
    
    # 2. Cari skor (KLASIFIKASI: X)
    score_match = re.search(r"KLASIFIKASI[:\- ]*([0-4])", text, re.IGNORECASE)
    
    # 3. Jika tidak ditemukan, coba cari angka tunggal (Plan B)
    if not score_match:
        score_match = re.search(r"\b([0-4])\b", text)

    score = None
    if score_match:
        try:
            # Mengubah ke integer
            score = int(score_match.group(1))
        except ValueError:
            # Jika tidak bisa diubah (meski regex sudah menjamin)
            score = None 

    # 4. Cari Alasan (ALASAN: ...)
    reason_match = re.search(r"ALASAN[:\-]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    
    # Gunakan alasan yang diparsing, atau seluruh teks jika parsing alasan gagal
    reason = reason_match.group(1).strip() if reason_match else text

    # Jika skor gagal diparsing (tetap None), beri alasan spesifik
    if score is None:
        reason = f"Parsing skor gagal. Output model tidak mengikuti format KLASIFIKASI: X. Raw: {text}"

    return score, reason


# ========================= SESSION INIT =========================
for key, default in {
    "page": "input",
    "results": [],
    "nama": "",
    "processing_done": False
}.items():
    st.session_state.setdefault(key, default)


# ========================= PAGE INPUT =========================
if st.session_state.page == "input":
    st.title("🎥 AI-Powered Interview Assessment System")
    st.write("Upload **5 video interview** lalu klik mulai analisis.")

    with st.form("upload_form"):
        nama = st.text_input("Nama Pelamar")
        uploaded = st.file_uploader(
            "Upload 5 Video (1 → 5)",
            type=["mp4", "mov", "mkv", "webm", "wav"],
            accept_multiple_files=True
        )
        submit = st.form_submit_button("Mulai Proses Analisis")

    if submit:
        if not nama:
            st.error("Nama wajib diisi.")
        elif not uploaded or len(uploaded) != 5:
            st.error("Harap upload tepat 5 video.")
        else:
            st.session_state.nama = nama
            st.session_state.uploaded = uploaded
            st.session_state.results = []
            st.session_state.page = "result"
            st.session_state.processing_done = True
            st.rerun()


# ========================= PAGE RESULT =========================
if st.session_state.processing_done and st.session_state.page == "result":
    st.title("📋 Hasil Penilaian Interview")
    st.write(f"**Nama Pelamar:** {st.session_state.nama}")

    progress = st.empty()

    if len(st.session_state.results) == 0:
        for idx, vid in enumerate(st.session_state.uploaded):
            progress.info(f"Memproses Video {idx+1}...")

            bytes_data = vid.read()
            transcript = transcribe_via_hf(bytes_data)
            # prompt = prompt_for_classification(INTERVIEW_QUESTIONS[idx], transcript)
            # raw_output = phi3_api(prompt)
            raw_output = llm_api(INTERVIEW_QUESTIONS[idx], transcript)
            score, reason = parse_model_output(raw_output)

            st.session_state.results.append({
                "question": INTERVIEW_QUESTIONS[idx],
                "transcript": transcript,
                "score": score,
                "reason": reason,
                "raw_model": raw_output
            })

            progress.success(f"Video {idx+1} selesai ✔")

    scores = [r["score"] for r in st.session_state.results if r["score"] is not None]

    if len(scores) == 5:
        final_score = sum(scores) / 5
        st.markdown(f"### ⭐ Skor Akhir: **{final_score:.2f} / 4**")
    else:
        st.error("Skor tidak semua berhasil diproses. Cek raw output model.")

    st.markdown("---")

    for i, r in enumerate(st.session_state.results):
        st.subheader(f"🎬 Video {i+1}")
        st.write(f"**Pertanyaan:** {r['question']}")
        st.write(f"**Transkrip:** {r['transcript']}")
        st.write(f"**Skor:** {r['score']}")
        st.write(f"**Alasan:** {r['reason']}")

        with st.expander("Raw Output Model"):
            st.code(r["raw_model"])

        st.markdown("---")

    if st.button("🔙 Kembali"):
        st.session_state.page = "input"
        st.session_state.processing_done = False
        st.session_state.results = []
        st.session_state.nama = ""
        st.rerun()
