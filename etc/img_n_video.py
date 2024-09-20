import ffmpeg
import cv2
import os
from pathlib import Path
from PIL import Image
import logging
from typing import List, Tuple, Dict, Any, Union, Optional

def convert_time_to_seconds(time_str):
    """Convert mi:ss format to seconds."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def extract_frames(video_path, start_time, duration):

    # Convert start time if it's in mi:ss format
    if isinstance(start_time, str):
        start_time = convert_time_to_seconds(start_time)

    # 비디오 파일 열기
    video = cv2.VideoCapture(video_path)
    
    # 비디오의 FPS(초당 프레임 수) 가져오기
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # 시작 프레임 계산
    start_frame = int(start_time * fps)
    
    # 종료 프레임 계산
    end_frame = int((start_time + duration) * fps)
    
    # 비디오의 시작 프레임으로 이동
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 현재 프레임 번호
    current_frame = start_frame
    
    # 출력 디렉토리 생성
    output_dir = "extracted_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    while current_frame < end_frame:
        
        # 프레임 읽기
        ret, frame = video.read()
        
        if not ret:
            break
        
        # 프레임을 이미지로 저장
        output_path = os.path.join(output_dir, f"frame_{(current_frame - start_frame):06d}.jpg")
        cv2.imwrite(output_path, frame)
        
        current_frame += 1
    
    # 비디오 파일 닫기
    video.release()
    
    print(f"프레임 추출 완료: {current_frame - start_frame}개의 프레임이 {output_dir} 디렉토리에 저장되었습니다.")

# 함수 사용 예:
# 
def create_video_from_images_and_audio(mp4_path, start, duration, images_path, output_path):

    # Convert start time if it's in mi:ss format
    if isinstance(start, str):
        start = convert_time_to_seconds(start)

    # Create a directory to store temporary files
    temp_dir = Path("temp_files")
    temp_dir.mkdir(exist_ok=True)
    
    # Extract audio from the original video
    audio_path = temp_dir / "audio.aac"
    audio_extracted = True

    try:
        (
            ffmpeg
            .input(mp4_path, ss=start, t=duration)
            .output(str(audio_path), acodec='aac', vn=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('Error extracting audio:')
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        audio_extracted = False
    
    # Find image files (jpg and png)
    image_files = sorted(Path(images_path).glob("*.jpg")) + sorted(Path(images_path).glob("*.png"))
    if not image_files:
        raise ValueError("No jpg or png images found in the specified directory.")
    
    # Get the frame rate of the original video
    probe = ffmpeg.probe(mp4_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    frame_rate = eval(video_stream['r_frame_rate'])  # Convert to a calculable value
    
    # Rename image files sequentially and move them to the temporary directory
    for idx, image_file in enumerate(image_files):
        new_name = temp_dir / f"{idx:04d}{image_file.suffix}"
        os.rename(image_file, new_name)
    
    # Convert images to video
    image_video_path = temp_dir / "image_video.mp4"

    try:
        (
            ffmpeg
            .input(str(temp_dir / "%04d.png"), framerate=frame_rate)
            .output(str(image_video_path), vcodec='libx264', pix_fmt='yuv420p')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('Error creating video from images:')
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e
    
    # Create the final video
    video = ffmpeg.input(str(image_video_path))

    if audio_extracted:
        audio = ffmpeg.input(str(audio_path))
        output_args = {
            'vcodec': 'copy',  # Copy the video stream as-is
            'acodec': 'aac',   # Encode the audio as AAC (in case the original isn't AAC)
            'shortest': None
        }
        inputs = [video, audio]
    else:
        output_args = {
            'vcodec': 'copy',  # Copy the video stream as-is
        }
        inputs = [video]

    (
        ffmpeg
        .output(*inputs, output_path, **output_args)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Delete temporary files
    for file in temp_dir.glob("*"):
        file.unlink()
    temp_dir.rmdir()

    print(f"Video creation complete: {output_path}")

# 함수 사용 예:
# 
def append_audio(mp4_path, start, duration, target_mp4_path, output_path):

    # Convert start time if it's in mi:ss format
    if isinstance(start, str):
        start = convert_time_to_seconds(start)

    # Create a directory to store temporary files
    temp_dir = Path("temp_files")
    temp_dir.mkdir(exist_ok=True)
    
    # Extract audio from the original video
    audio_path = temp_dir / "audio.aac"

    try:
        (
            ffmpeg
            .input(mp4_path, ss=start, t=duration)
            .output(str(audio_path), acodec='aac', vn=None)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print('Error extracting audio:')
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e    
    
    # Create the final video
    video = ffmpeg.input(str(target_mp4_path))

    audio = ffmpeg.input(str(audio_path))
    output_args = {
        'vcodec': 'copy',  # Copy the video stream as-is
        'acodec': 'aac',   # Encode the audio as AAC (in case the original isn't AAC)
        'shortest': None
    }
    inputs = [video, audio]
    
    (
        ffmpeg
        .output(*inputs, output_path, **output_args)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Delete temporary files
    for file in temp_dir.glob("*"):
        file.unlink()
    temp_dir.rmdir()

    print(f"Video creation complete: {output_path}")

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