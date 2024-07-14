import yt_dlp
from yt_dlp.utils import download_range_func
import cv2
import os
import numpy as np
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
from langdetect import detect, DetectorFactory

# 언어 감지기의 동작을 일관되게 하기 위해 시드 값 설정
DetectorFactory.seed = 0

def get_language_code(text):
    try:
        # 입력된 텍스트의 언어 감지
        language_code = detect(text)
        return language_code
    except Exception as e:
        return str(e)

def youtube_to_numpy(youtube_url, start_time, duration):
    ydl_opts = {'format': 'best[ext=mp4]', 'nocheckcertificate': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        video_url = info['url']
        fps = info['fps']

    cap = cv2.VideoCapture(video_url)
    
    # 시작 위치로 이동
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    
    # 첫 프레임을 읽어 배열의 shape을 결정합니다
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("비디오를 읽을 수 없습니다.")
    
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    frame_shape = first_frame_rgb.shape
    
    # 필요한 프레임 수 계산
    total_frames = int(duration * fps)
    
    # 미리 할당된 NumPy 배열 생성
    numpy_array = np.empty((total_frames, *frame_shape), dtype=np.uint8)
    numpy_array[0] = first_frame_rgb
    
    frame_count = 1
    
    for i in range(1, total_frames):
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

from enum import Enum

class Position(Enum):
    CENTER = 'center'
    BOTTOM = 'bottom'
    TOP = 'top'

__DEFAULT_FONT_PATH = 'C:\\Windows\\Fonts\\gulim.ttc'

def is_tuple_int_int(var):
    return isinstance(var, tuple) and len(var) == 2 and all(isinstance(i, int) for i in var)

def download_youtube(youtube_url):
    ydl_opts = {
        #'format': 'best[ext=mp4]',
        #'outtmpl': 'temp_video.%(ext)s',
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': '%(id)s.%(ext)s',
        'nocheckcertificate': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(youtube_url, download=True)

def download_youtube_segment(youtube_url, segments, text_configs=None):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': 'temp_video.%(ext)s',
        'nocheckcertificate': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        temp_filename = ydl.prepare_filename(info)

    try:
        for i, (start_time, duration, output_filename) in enumerate(segments):
            try:
                cap = cv2.VideoCapture(temp_filename)
                cap.set(cv2.CAP_PROP_POS_MSEC, int(start_time * 1000))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                
                if text_configs and i < len(text_configs):
                    text_config = text_configs[i]
                    text = text_config.get('text', 'COPIED')
                    font_path = text_config.get('font_path', __DEFAULT_FONT_PATH) # 한글 폰트 파일 경로
                    font_size = text_config.get('font_size', 48)
                    font = ImageFont.truetype(font_path, font_size)
                    color = text_config.get('color', (0, 0, 255))
                    
                    position = text_config.get('position', 'center')

                    if not is_tuple_int_int(position):

                        # PIL Image로 텍스트 크기 계산
                        pil_img = Image.new('RGB', (width, height))
                        draw = ImageDraw.Draw(pil_img)

                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        text_x = (width - text_width) // 2

                        match position:
                            case Position.CENTER:
                                text_y = (height - text_height) // 2
                            case Position.BOTTOM:
                                text_y = (height - text_height) - 10
                            case Position.TOP:
                                text_y = 10

                        position = (text_x, text_y)

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame_count >= duration * fps:
                        break
                    
                    if text_configs:
                        # OpenCV 프레임을 PIL Image로 변환
                        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_frame)
                        
                        # 텍스트 그리기
                        draw.text(position, text, font=font, fill=color[::-1])  # RGB to BGR
                        
                        # PIL Image를 다시 OpenCV 프레임으로 변환
                        frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                    
                    out.write(frame)
                    frame_count += 1
                
                print('type(frame) : ', type(frame))
                cap.release()
                out.release()
                print(f"Video segment saved as {output_filename}")
                
            except Exception as e:
                print(f"Error processing segment {output_filename}: {str(e)}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)