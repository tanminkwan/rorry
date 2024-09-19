"""
pip install insightface
pip install onnxruntime #for CPU-only
pip install onnxruntime-gpu #For GPU
pip uninstall opencv-python-headless
pip install opencv-contrib-python-headless

buffalo_l download from : 
https://github.com/deepinsight/insightface/releases

unzip buffalo_l.zip on C:\\Users\\<user>\\.insightface\\models\\buffalo_l.
"""
import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

assert insightface.__version__ >= '0.7'

class FaceSwapper:
    def __init__(self, model_name='buffalo_l', ctx_id=-1, det_size=(640, 640)):
        # 얼굴 분석 모델 초기화
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        # 얼굴 교체 모델 로드
        self.swapper = insightface.model_zoo.get_model(
            'C:\\pypjt\\env\\inswapper_128.onnx', download=True, download_zip=True
        )
        # 소스 얼굴 초기화
        self.source_face = None

    def set_source_face(self, img, face_index=0):
        """
        이미지에서 소스 얼굴을 설정합니다.
        img: 이미지 파일 경로나 numpy.ndarray 이미지
        face_index: 선택할 얼굴의 인덱스 (기본값: 0)
        """
        # 이미지 로드 (파일 경로 또는 ndarray 처리)
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                print(f"이미지를 로드할 수 없습니다: {img}")
                return False
        elif not isinstance(img, np.ndarray):
            print("유효한 이미지 또는 이미지 경로를 입력해 주세요.")
            return False

        # 얼굴 검출
        faces = self.app.get(img)
        if len(faces) == 0:
            print("소스 이미지에서 얼굴을 감지하지 못했습니다.")
            return False
        # 얼굴을 x 좌표 기준으로 정렬
        faces = sorted(faces, key=lambda x: x.bbox[0])
        if face_index >= len(faces):
            print(f"소스 얼굴 인덱스가 범위를 벗어났습니다. 총 감지된 얼굴 수: {len(faces)}")
            return False
        # 소스 얼굴 설정
        self.source_face = faces[face_index]
        print(f"소스 얼굴이 설정되었습니다. 인덱스: {face_index}")
        return True

    def swap_faces_in_image(self, img):
        """
        ndarray 이미지를 입력으로 받아 얼굴 교체를 수행하고, 결과 이미지를 반환합니다.
        """
        if self.source_face is None:
            print("소스 얼굴이 설정되지 않았습니다. 먼저 set_source_face 메서드를 호출하여 소스 얼굴을 설정하세요.")
            return None
        # 얼굴 검출
        faces = self.app.get(img)
        if len(faces) == 0:
            print("대상 이미지에서 얼굴이 감지되지 않았습니다.")
            return None
        # 얼굴을 x 좌표 기준으로 정렬
        faces = sorted(faces, key=lambda x: x.bbox[0])
        # 얼굴 교체 수행
        res = img.copy()
        for face in faces:
            res = self.swapper.get(res, face, self.source_face, paste_back=True)
        return res

    def extract_and_swap_faces_in_image(self, img):
        """
        ndarray 이미지를 입력으로 받아 개별 얼굴을 교체한 이미지를 반환합니다.
        """
        if self.source_face is None:
            print("소스 얼굴이 설정되지 않았습니다. 먼저 set_source_face 메서드를 호출하여 소스 얼굴을 설정하세요.")
            return None
        # 얼굴 검출
        faces = self.app.get(img)
        if len(faces) == 0:
            print("대상 이미지에서 얼굴이 감지되지 않았습니다.")
            return None
        # 얼굴을 x 좌표 기준으로 정렬
        faces = sorted(faces, key=lambda x: x.bbox[0])
        # 개별 얼굴 교체 및 추출
        res = []
        for face in faces:
            _img, _ = self.swapper.get(img, face, self.source_face, paste_back=False)
            res.append(_img)
        if len(res) == 0:
            print("교체된 얼굴이 없습니다.")
            return None
        res = np.concatenate(res, axis=1)
        return res

if __name__ == '__main__':
    # FaceSwapper 클래스 인스턴스 생성
    face_swapper = FaceSwapper(det_size=(320, 320))
    
    # 소스 얼굴 이미지 로드
    source_img = cv2.imread('./kim.jpg')
    #source_img = ins_get_image('t1')  # 또는 cv2.imread('source.jpg')
    
    # 소스 얼굴 설정 (face_index는 선택 사항)
    success = face_swapper.set_source_face(source_img, face_index=0)
    if not success:
        print("소스 얼굴 설정에 실패했습니다.")
        exit()
    
    # 대상 이미지 로드 ('t2' 이미지를 사용하거나 경로를 직접 지정할 수 있습니다)
    target_img = ins_get_image('t1')  # 또는 cv2.imread('target.jpg')
    
    # 얼굴 교체 수행 (ndarray 이미지를 입력으로 받아 결과를 ndarray로 반환)
    swapped_img = face_swapper.swap_faces_in_image(target_img)
    if swapped_img is not None:
        # 결과 이미지 저장 또는 추가 처리
        cv2.imwrite('./t2_swapped.jpg', swapped_img)
        print("얼굴 교체가 완료되었습니다. 결과 이미지를 저장했습니다.")
    else:
        print("얼굴 교체에 실패했습니다.")
    
    # 개별 얼굴 교체 및 추출
    extracted_swapped_img = face_swapper.extract_and_swap_faces_in_image(target_img)
    if extracted_swapped_img is not None:
        # 결과 이미지 저장 또는 추가 처리
        cv2.imwrite('./t2_swapped_individual.jpg', extracted_swapped_img)
        print("개별 얼굴 교체가 완료되었습니다. 결과 이미지를 저장했습니다.")
    else:
        print("개별 얼굴 교체에 실패했습니다.")
