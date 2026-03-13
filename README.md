# Candidate Speech Analyzer

## Deskripsi Singkat Proyek
Candidate Speech Analyzer adalah platform berbasis kecerdasan buatan untuk melakukan penilaian wawancara khusus machine learning secara otomatis. Sistem ini membantu perusahaan dan organisasi dalam melakukan screening kandidat dengan lebih efisien menggunakan teknologi AI.

## Fitur Utama
- 🎯 Analisis jawaban kandidat secara real-time
- 📊 Penilaian otomatis berdasarkan kriteria yang ditentukan
- 🎤 Dukungan untuk wawancara berbasis teks dan suara
- 🔒 Keamanan data kandidat yang terjamin
- 📝 Rekap output hasil wawancara

## Setup Environment

### Prasyarat
Pastikan sistem Anda telah memiliki:
- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Virtual environment (disarankan)

### Langkah Instalasi

1. **Clone repository**
   ```bash
   git clone https://github.com/Capstone-Project-Asah-DC-01/AI_Interview_Assessment_System.git
   cd AI_Interview_Assessment_System
   ```

2. **Buat virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   # atau
   venv\Scripts\activate  # Untuk Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi environment variables**
   ```bash
   cp .env.example .env
   # Edit file .env sesuai dengan konfigurasi Anda
   ```

5. **Instalasi FFMPEG**
   Windows Only:
   ```bash
   kunjungi web https://ffmpeg.org/.
   download sesuai dengan device
   ekstrak (unzip) file .zip
   disarankan penempatan file pada path => C:\Program Files\ffmpeg

   Copy path bin yang ada di ffmpeg, contoh path => C:\Program Files\ffmpeg\bin

   Masuk ke dalam dalam kontrol panel, pada edit the system environment variables.
   set Environment Variables
   pada box variables cari variable "PATH",
   Tekan tombol Edit, kemudian buat baru path dan tambahkan path C:\Program Files\ffmpeg\bin
   kemudian tekan tombol oke
   
   ```

## Tautan Model ML
   https://colab.research.google.com/drive/1PCqx7bCYuq78D9WHxIlNQgWOrg-2NtYZ?usp=sharing

## **Cara menjalankan aplikasi**
   ```bash
   Buka powershell pada code editor
   kemudian masuk ke virtual environment yang sudah dibuat dengan cara
   Scripts\activate
   kemudian jalankan dengan kode berikut:
   streamlit run interview_assessment.py
   atau
   streamlit run interview_iterative_version.py
   atau
   streamlit run interview_hybrid_version.py
   ```
## 🙏 Acknowledgments

- Terima kasih kepada semua kontributor
- Inspirasi dari berbagai open-source interview assessment tools
- Dukungan dari komunitas AI dan ML

---




