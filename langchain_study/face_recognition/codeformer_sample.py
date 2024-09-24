import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

"""
Create Virtual Env :
    python -m venv face
Download CodeFormer pjt :
    https://github.com/sczhou/CodeFormer
    cd CodeFormer
    pip install -r requirements.txt
    python basicsr/setup.py develop
Download CodeFormer model:
    https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
    locate it in specific directory
Download Face relevant models :
    https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth
    https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth
    locate it in ~\CodeFormer\weights\facelib
"""

def main():
    # 모델 로드
    #GPU model = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).cuda()
    model = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256'])
    
    ckpt_path = "C:\\pypjt\\env\\codeformer.pth"
    #GPU checkpoint = torch.load(ckpt_path)['params_ema']
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location='cpu')['params_ema']

    model.load_state_dict(checkpoint)
    model.eval()

    # 입력 이미지 로드
    img_path = "C:\\pypjt\\face\\test_hanni2.jpg"
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # 얼굴 검출 및 정렬
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50', # detection model
        save_ext='png',
        use_parse=True,
        #GPU device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device=torch.device('cpu')
    )
    face_helper.read_image(img)
    face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    face_helper.align_warp_face()

    # 얼굴 복원
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        #GPU cropped_face_t = cropped_face_t.unsqueeze(0).cuda()
        cropped_face_t = cropped_face_t.unsqueeze(0)

        try:
            with torch.no_grad():
                output = model(cropped_face_t, w=0.5, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            #torch.cuda.empty_cache()
        except RuntimeError as error:
            print(f'Error: {error}')
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

    # 결과 저장
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    
    # 최종 이미지 크기 조정 (원본 크기로)
    restored_img = cv2.resize(restored_img, (w, h))

    save_path = 'restored_image.png'
    cv2.imwrite(save_path, restored_img)

if __name__ == '__main__':
    main()