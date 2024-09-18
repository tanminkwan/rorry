import requests
import io
import base64
from PIL import Image

def get_swapped_face(source_bytes: bytes, 
                     target_bytes: bytes, 
                     sp_url: str="http://127.0.0.1:7860/reactor/image",
                     device: str="GPU") -> bytes:

    source_image = base64.b64encode(source_bytes).decode('utf-8')
    target_image = base64.b64encode(target_bytes).decode('utf-8')

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "source_image": source_image,
        #"source_image": "",
        "target_image": target_image,
        #"target_image": "",
        "source_faces_index": [0],
        "face_index": [0],
        "upscaler": "None",
        "scale": 1,
        "upscale_visibility": 1,
        "face_restorer": "CodeFormer",
        "restorer_visibility": 1,
        "restore_first": 1,
        "model": "inswapper_128.onnx", #.\models\insightface
        "gender_source": 0,
        "gender_target": 0,
        "save_to_file": 0,
        #"result_file_path": "",
        "device": device,
        "mask_face": 0,
        "select_source": 0,
        #"face_model": "elena.safetensors",
        #"source_folder": "C:/faces",
        #"random_image": 1,
        "upscale_force": 0
    }

    response = requests.post(url=sp_url, headers=headers, json=payload)
    response.raise_for_status()
    
    # 결과 이미지 저장
    result = response.json()
    image_data = base64.b64decode(result['image'])
    return io.BytesIO(image_data)

if __name__ == "__main__":
    # 사용 예시
    source_image_path = "./faces/김태희.jpg"  # 소스 얼굴 이미지 경로
    target_image_path = "./faces/bk.goldengirls01.jpg"  # 대상 이미지 경로
    output_path = "face_swapped_result.png"  # 결과 이미지 저장 경로

    # 이미지 로드 및 base64 인코딩
    with open(source_image_path, "rb") as source_file:
        source_bytes = source_file.read()
        
    with open(target_image_path, "rb") as target_file:
        target_bytes = target_file.read()

    out_bytes = get_swapped_face(source_bytes, target_bytes)

    image = Image.open(out_bytes)
    image.save(output_path)
