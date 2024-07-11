import yt_dlp
import cv2
import numpy as np
import functools

def youtube2mp4(youtube_url, output_file, max_frames=2000, fps=30):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # YouTube에서 비디오 다운로드
            video_array = youtube_to_numpy(youtube_url, max_frames)
            
            # 함수 실행 (비디오 처리)
            processed_video = func(video_array, *args, **kwargs)
            
            # 처리된 비디오를 MP4로 저장
            save_numpy_to_mp4(processed_video, output_file, fps)
            
            return processed_video
        return wrapper
    return decorator

def youtube_to_numpy(youtube_url, max_frames=500):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        video_url = info['url']

    cap = cv2.VideoCapture(video_url)
    
    # 첫 프레임을 읽어 배열의 shape을 결정합니다
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("비디오를 읽을 수 없습니다.")
    
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    frame_shape = first_frame_rgb.shape
    
    # 미리 할당된 NumPy 배열 생성
    numpy_array = np.empty((max_frames, *frame_shape), dtype=np.uint8)
    numpy_array[0] = first_frame_rgb
    
    frame_count = 1
    
    for i in range(1, max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        numpy_array[i] = frame_rgb
        frame_count += 1
    
    cap.release()

    # 실제로 읽은 프레임 수만큼 배열 크기 조정
    numpy_array = numpy_array[:frame_count]

    return numpy_array

def save_numpy_to_mp4(numpy_array, output_filename, fps=30):
    if len(numpy_array.shape) != 4:
        raise ValueError("NumPy 배열은 (프레임, 높이, 너비, 채널) 형태여야 합니다.")
    
    height, width = numpy_array.shape[1:3]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    for frame in numpy_array:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    
    print(f"비디오가 {output_filename}로 저장되었습니다.")

# 데코레이터 사용 예시
@youtube2mp4(youtube_url="https://www.youtube.com/watch?v=54ojPbRb9S4", output_file="output_video.mp4")
def add_copied_text(video_array):
    height, width = video_array.shape[1:3]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 5
    text = "COPIED"
    
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    for i in range(len(video_array)):
        frame = video_array[i].copy()
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
        video_array[i] = frame
    
    return video_array

# 함수 실행 (데코레이터에 의해 자동으로 YouTube에서 다운로드하고 MP4로 저장됨)
processed_video = add_copied_text()
print(f"처리된 비디오 배열 shape: {processed_video.shape}")