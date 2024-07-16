import yt_dlp
import cv2
import os
import pyttsx3
import tempfile
import numpy as np
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
from langdetect import detect, DetectorFactory
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
import ffmpeg
import logging
import shlex
from typing import List, Tuple, Dict, Any, Union, Optional

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 언어 감지기의 동작을 일관되게 하기 위해 시드 값 설정
DetectorFactory.seed = 0

def get_language_code(text):
    try:
        # 입력된 텍스트의 언어 감지
        language_code = detect(text)
        return language_code
    except Exception as e:
        return str(e)

def get_youtube_captions(id,language='ko'):
    
    try:
        scripts = YouTubeTranscriptApi.get_transcript(id, languages=[language])
    except NoTranscriptFound:
        scripts = []
    return scripts

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

def add_text_to_image(image_path, output_path, text='text', position=Position.CENTER, font_size=72, font_color=(255, 255, 255), line_spacing=1.4):
    # 이미지 열기
    with Image.open(image_path) as img:
        # 이미지 크기 가져오기
        width, height = img.size
        
        # 그리기 객체 생성
        draw = ImageDraw.Draw(img)
        
        # 폰트 설정 (기본 폰트 사용, 실제 사용시 폰트 경로를 지정하는 것이 좋습니다)
        try:
            font = ImageFont.truetype(__DEFAULT_FONT_PATH, font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # 텍스트를 여러 줄로 나누는 함수
        def wrap_text(text, font, max_width):
            lines = []
            words = text.split()
            current_line = words[0]
            for word in words[1:]:
                test_line = current_line + " " + word
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            lines.append(current_line)
            return lines
        
        # 텍스트를 여러 줄로 나누기
        max_width = width - 20  # 양쪽에 10픽셀의 여백
        lines = wrap_text(text, font, max_width)
        
        # 전체 텍스트 높이 계산
        line_height = draw.textbbox((0, 0), 'Aj', font=font)[3] - draw.textbbox((0, 0), 'Aj', font=font)[1]
        total_text_height = line_height * len(lines) * line_spacing
        
        # 텍스트 위치 결정
        if isinstance(position, tuple):
            text_position = position
        elif isinstance(position, Position):
            if position == Position.CENTER:
                text_position = ((width - max_width) // 2, (height - total_text_height) // 2)
            elif position == Position.BOTTOM:
                text_position = ((width - max_width) // 2, height - total_text_height - 10)
            elif position == Position.TOP:
                text_position = ((width - max_width) // 2, 10)
        else:
            text_position = (10, 10)  # 기본값은 좌상단
        
        # 텍스트 추가
        y_text = text_position[1]
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x_text = text_position[0] + (max_width - line_width) // 2  # 각 줄을 중앙 정렬
            draw.text((x_text, y_text), line, font=font, fill=font_color)
            y_text += line_height * line_spacing
        
        # 결과 저장
        img.save(output_path)
        
def is_tuple_int_int(var):
    return isinstance(var, tuple) and len(var) == 2 and all(isinstance(i, int) for i in var)

def download_youtube(youtube_url, job_id=''):
    ydl_opts = {
        #'format': 'best[ext=mp4]',
        #'outtmpl': 'temp_video.%(ext)s',
        #'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'format': 'best[ext=mp4]/best',
        'outtmpl': job_id+'_'+'%(id)s.%(ext)s',
        'nocheckcertificate': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
        
    file_path = os.path.abspath(filename)
    return file_path, os.path.basename(file_path)

def slice_video(file_name: str, segments: List[Tuple[float, float, str]], height: Optional[int] = None, width: Optional[int] = None) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    
    for start, duration, output_name in segments:
        segment_info = {
            "output_name": output_name,
            "start": start,
            "duration": duration,
            "success": False,
            "error": None
        }
        
        try:
            input_video = ffmpeg.input(file_name, ss=start, t=duration)
            
            # 비디오 스트림 처리
            video = input_video.video
            if height is not None or width is not None:
                video = video.filter('scale', width=width, height=height)
            
            # 오디오 스트림 처리
            audio = input_video.audio

            # 비디오와 오디오 스트림 결합
            output = ffmpeg.output(video, audio, output_name)
            
            output.overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
            logging.info(f"Successfully created segment: {output_name}")
            segment_info["success"] = True
        except ffmpeg.Error as e:
            error_message = e.stderr.decode()
            logging.error(f"Error occurred while processing {output_name}: {error_message}")
            segment_info["error"] = error_message
        
        results.append(segment_info)
    
    return results

def draw_text_cv2(video_file_name: str, segments: List[Tuple[float, float, Dict[str, Union[str, int, Tuple[int, int, int]]]]]) -> None:
    logging.info(f"Starting text overlay process for {video_file_name}")
    cap = cv2.VideoCapture(video_file_name)
    
    # 비디오 속성 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logging.info(f"Video properties: FPS={fps}, Width={width}, Height={height}")
    
    # 임시 출력 파일 이름 생성
    file_name, file_extension = os.path.splitext(video_file_name)
    temp_output = f"{file_name}_temp{file_extension}"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_count / fps
        
        for start_time, duration, text_config in segments:

            if start_time <= current_time < start_time + duration:
                logging.debug(f"Adding text overlay at time {current_time:.2f}s")
                # OpenCV 프레임을 PIL Image로 변환
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_frame)
                
                # 텍스트 설정 추출
                text = text_config.get('text', 'COPIED')
                font_path = text_config.get('font_path', __DEFAULT_FONT_PATH)
                font_size = text_config.get('font_size', 48)
                font = ImageFont.truetype(font_path, font_size)
                color = text_config.get('color', (255, 0, 0))  # RGB
                position = text_config.get('position', Position.CENTER)

                # 텍스트 위치 계산
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                if isinstance(position, tuple):
                    text_position = position
                elif isinstance(position, Position):
                    if position == Position.CENTER:
                        text_position = ((width - text_width) // 2, (height - text_height) // 2)
                    elif position == Position.BOTTOM:
                        text_position = ((width - text_width) // 2, height - text_height - 10)
                    elif position == Position.TOP:
                        text_position = ((width - text_width) // 2, 10)
                else:
                    text_position = (10, 10)  # 기본값은 좌상단

                # 텍스트 그리기
                draw.text(text_position, text, font=font, fill=color)
                
                # OpenCV 프레임으로 다시 변환
                frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
                
                break  # 이 프레임에 대해 다른 세그먼트 확인 중지
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    # 원본 파일을 새 파일로 교체
    os.replace(temp_output, video_file_name)
    
    logging.info(f"Text overlay completed for {video_file_name}")

def draw_text(video_file_name: str, segments: List[Tuple[float, float, Dict[str, Union[str, int, Tuple[int, int, int], Position]]]]) -> None:
    logging.info(f"Starting text overlay process for {video_file_name}")

    # 입력 비디오 스트림
    input_stream = ffmpeg.input(video_file_name)

    # 각 세그먼트에 대한 필터 생성
    filter_complex = []

    for i, (start_time, duration, text_config) in enumerate(segments):
        text = text_config['text']
        font_size = text_config.get('font_size', 48)
        font_color = text_config.get('font_color', (255, 255, 255))  # 기본값은 흰색
        position = text_config.get('position', Position.CENTER)
        fontfile = text_config.get('fontfile', '/Windows/Fonts/batang.ttc')

        # 색상을 16진수 문자열로 변환
        color_hex = '0x%02x%02x%02x' % font_color

        # 위치 설정
        if position == Position.CENTER:
            x = '(w-tw)/2'
            y = '(h-th)/2'
        elif position == Position.BOTTOM:
            x = '(w-tw)/2'
            y = 'h-th-10'
        elif position == Position.TOP:
            x = '(w-tw)/2'
            y = '10'
        else:
            x = '10'
            y = '10'

        # 필터 문자열 생성
        filter_str = (
            f"drawtext=fontfile='{fontfile}':text='{text}':fontsize={font_size}:fontcolor={color_hex}:"
            f"x={x}:y={y}:enable='between(t,{start_time},{start_time+duration})'"
        )

        # box 설정이 있는 경우 추가
        if 'box' in text_config:
            filter_str += f":box={text_config['box']}"
        if 'boxcolor' in text_config:
            boxcolor = text_config['boxcolor']
            if isinstance(boxcolor, tuple):
                boxcolor_hex = '0x%02x%02x%02x%02x' % (boxcolor[0], boxcolor[1], boxcolor[2], boxcolor[3] if len(boxcolor) > 3 else 255)
            else:
                boxcolor_hex = boxcolor
            filter_str += f":boxcolor={boxcolor_hex}"
        if 'boxborderw' in text_config:
            filter_str += f":boxborderw={text_config['boxborderw']}"

        filter_complex.append(filter_str)

    # 모든 필터를 연결
    filter_complex_str = ','.join(filter_complex)

    # 출력 파일 이름 생성
    file_name, file_extension = os.path.splitext(video_file_name)
    output_file = f"{file_name}_t{file_extension}"

    # 출력 스트림 설정
    output_stream = ffmpeg.output(
        input_stream,
        output_file,
        vf=filter_complex_str,
        acodec='copy'  # 오디오는 그대로 복사
        #vcodec='libx264',  # H.264 코덱 사용
        #preset='medium',  # 인코딩 속도와 품질의 균형
        #crf=23  # 품질 설정 (낮을수록 품질 높음, 18-28 권장)
    )

    # ffmpeg 실행
    logging.info("Starting ffmpeg process")
    try:
        ffmpeg.run(output_stream, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    logging.info(f"Text overlay completed. Output saved as {output_file}")

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

def _create_audio_from_text(text, output_path):
    engine = pyttsx3.init(driverName='sapi5')
    engine.setProperty('rate', 150)
    engine.save_to_file(text, output_path)
    engine.runAndWait()

def create_video_with_audio_and_image(image_path, output_path, text='COPIED', width=640, height=360):
    # 음성 파일 생성
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
    
    _create_audio_from_text(text, temp_audio_path)
    
    # 음성 파일의 길이 확인
    probe = ffmpeg.probe(temp_audio_path)
    audio_duration = float(probe['streams'][0]['duration']) + 0.5
    
    # 이미지를 비디오로 변환
    video_stream = ffmpeg.input(image_path, loop=1, t=audio_duration)
    video_stream = ffmpeg.filter(video_stream, 'scale', width, height)
    
    # 오디오 스트림 생성
    audio_stream = ffmpeg.input(temp_audio_path)
    
    # 비디오와 오디오 결합
    output = ffmpeg.output(video_stream, audio_stream, output_path,
                           vcodec='libx264',
                           acodec='aac',
                           pix_fmt='yuv420p',
                           r=24)
    
    #ffmpeg.run(output)
    # FFmpeg 명령 실행 및 오류 디버깅
    try:
        ffmpeg.run(output, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    
    # 임시 오디오 파일 삭제
    os.remove(temp_audio_path)

def get_video_properties(file):
    probe = ffmpeg.probe(file)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    
    if video_stream is None:
        raise Exception(f"No video stream found in {file}")

    return {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'r': eval(video_stream['r_frame_rate']),
        'pix_fmt': video_stream.get('pix_fmt', 'yuv420p'),
        'vcodec': video_stream['codec_name']
    }

def concatenate_videos(input_files, output_file, width=None, height=None):
    if not input_files:
        raise Exception("No input files provided")

    # 첫 번째 비디오의 속성 가져오기
    first_video_props = get_video_properties(input_files[0])
    
    # width와 height가 제공되지 않은 경우 첫 번째 비디오의 값 사용
    target_width = width if width is not None else first_video_props['width']
    target_height = height if height is not None else first_video_props['height']
    
    input_streams = []
    
    for file in input_files:
        # 비디오 스트림 추가 (목표 크기로 변환)
        video = ffmpeg.input(file)['v']
        video = ffmpeg.filter(video, 'scale', width=target_width, height=target_height)
        video = ffmpeg.filter(video, 'fps', fps=first_video_props['r'])
        video = ffmpeg.filter(video, 'setsar', sar='1')
        input_streams.append(video)
        
        # 오디오 스트림이 있는 경우에만 추가
        probe = ffmpeg.probe(file)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        if audio_streams:
            input_streams.append(ffmpeg.input(file)['a'])
        else:
            # 오디오가 없는 경우 무음 추가
            video_duration = float(probe['format']['duration'])
            input_streams.append(ffmpeg.input(f'anullsrc=duration={video_duration}', f='lavfi')['a'])

    # 모든 스트림 연결
    joined = ffmpeg.concat(*input_streams, v=1, a=1)

    # 출력 파일 생성 (첫 번째 비디오의 속성 사용)
    output = ffmpeg.output(joined, output_file,
                           vcodec=first_video_props['vcodec'],
                           pix_fmt=first_video_props['pix_fmt'])

    # FFmpeg 명령 실행 및 오류 디버깅
    try:
        ffmpeg.run(output, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e