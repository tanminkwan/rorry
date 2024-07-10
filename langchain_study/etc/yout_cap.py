import yt_dlp
import cv2
import numpy as np

def youtube_to_numpy(youtube_url, max_frames=100):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        video_url = info['url']

    cap = cv2.VideoCapture(video_url)
    frames = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    numpy_array = np.array(frames)
    cap.release()

    return numpy_array

def save_numpy_to_mp4(numpy_array, output_filename, fps=30):
    # NumPy 배열의 차원 확인
    if len(numpy_array.shape) != 4:
        raise ValueError("NumPy 배열은 (프레임, 높이, 너비, 채널) 형태여야 합니다.")
    
    # 비디오 크기 가져오기
    height, width = numpy_array.shape[1:3]
    
    # VideoWriter 객체 생성
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    # 각 프레임을 비디오에 쓰기
    for frame in numpy_array:
        # RGB에서 BGR로 변환 (OpenCV는 BGR 형식을 사용)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    # VideoWriter 객체 해제
    out.release()
    
    print(f"비디오가 {output_filename}로 저장되었습니다.")

def add_copied_text(video_array):
    height, width = video_array.shape[1:3]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 5
    text = "COPIED"
    
    # 텍스트 크기 계산
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # 텍스트 위치 계산 (화면 중앙)
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # 각 프레임에 텍스트 추가
    for i in range(len(video_array)):
        frame = video_array[i].copy()
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
        video_array[i] = frame
    
    return video_array
# 사용 예시
youtube_url = "https://www.youtube.com/watch?v=54ojPbRb9S4"
video_array = youtube_to_numpy(youtube_url, max_frames=500)

print(f"Shape of the NumPy array: {video_array.shape}")
print(f"Data type of the NumPy array: {video_array.dtype}")

video_array = add_copied_text(video_array)
output_filename = "output_video.mp4"
save_numpy_to_mp4(video_array, output_filename, fps=30)
