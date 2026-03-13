import shutil
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    print(f"FFMPEG ditemukan di: {ffmpeg_path}")
else:
    print("FFMPEG TIDAK ditemukan di PATH.")