import youtube_dl
import cv2
import numpy as np

def get_youtube_stream(youtube_url):
    ydl_opts = {}
    ydl = youtube_dl.YoutubeDL(ydl_opts)
    info_dict = ydl.extract_info(youtube_url, download=False)
    formats = info_dict.get('formats', None)
    
    for f in formats:
        if f.get('format_note', None) == '720p':
            url = f.get('url', None)
            return url
    
    return formats[-1]['url']

def stream_to_numpy(youtube_url):
    stream_url = get_youtube_stream(youtube_url)
    cap = cv2.VideoCapture(stream_url)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        # 예시로 100프레임만 캡처
        if len(frames) >= 100:
            break
    
    cap.release()
    
    return np.array(frames)

# 사용 예시
youtube_url = "https://youtu.be/1bUy-1hGZpI?si=3WW1uLBAIc2kKEiD"  # 원하는 YouTube 비디오 URL로 변경하세요
video_array = stream_to_numpy(youtube_url)

print(f"Shape of video array: {video_array.shape}")