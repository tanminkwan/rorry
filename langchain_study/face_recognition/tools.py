import numpy as np
import os
import cv2
import torch
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.app.common import Face
from typing import Tuple, Optional, Literal
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY
from sklearn.metrics.pairwise import cosine_similarity
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer

assert insightface.__version__ >= '0.7'

#SWAPPER_MODLE = 'C:\\Users\\tanmi\\stable-diffusion-webui\\models\\insightface\\inswapper_128.onnx'
SWAPPER_MODLE = 'C:\\pypjt\\env\\inswapper_128.onnx'

#CodeFormer 모델 경로 설정
CODEFORMER_MODEL = "C:\\pypjt\\env\\codeformer.pth"

class FaceSwapper:
    def __init__(self, model_name='buffalo_l', ctx_id=-1, det_size=(640, 640), nms_thresh = 0.6):
        # 얼굴 분석 모델 초기화
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.app.det_model.nms_thresh = nms_thresh
        # 얼굴 교체 모델 로드
        self.swapper = insightface.model_zoo.get_model(
            SWAPPER_MODLE, download=True, download_zip=True
        )
        # 소스 얼굴 초기화
        self.source_face = None
        self.enhanced=False

    def set_source_face(self, img, face_index=0, enhanced=False):
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

        self.enhanced=enhanced
        
        return True

    def enhance_image(self, img):
        # 샤프닝
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
        
        # 대비 향상
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def swap_faces_in_image(self, img, draw_rectangle=False):
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
        print(f"faces count : {len(faces)}")
        # 얼굴 교체 수행
        res = img.copy()
        for face in faces:
            res = self.swapper.get(res, face, self.source_face, paste_back=True)
            # 얼굴 교체된 경우 사각형 그리기
            if draw_rectangle:
                bbox = face.bbox.astype(int)
                cv2.rectangle(res, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # 이미지 품질 향상
        if self.enhanced:
            res = self.enhance_image(res)

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

def restore_face(input_image: np.ndarray, use_gpu: bool=False, draw_rectangle: bool=True)->np.ndarray:
    """
    얼굴 복원 함수
    
    :param input_image: ndarray 타입의 입력 이미지
    :param model_path: CodeFormer 모델 파일 경로
    :param use_gpu: GPU 사용 여부 (기본값: False)
    :return: ndarray 타입의 복원된 이미지
    """
    # 모델 로드
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(device)
    
    checkpoint = torch.load(CODEFORMER_MODEL, weights_only=True, map_location=device)['params_ema']
    model.load_state_dict(checkpoint)
    model.eval()

    # 이미지 크기 저장
    h, w, _ = input_image.shape

    # 얼굴 검출 및 정렬
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device
    )
    face_helper.read_image(input_image)
    face_helper.get_face_landmarks_5(only_center_face=False, resize=512, eye_dist_threshold=5)
    #face_helper.get_face_landmarks_5(only_center_face=False, resize=1024, eye_dist_threshold=5)
    face_helper.align_warp_face()

    # 얼굴 복원
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = model(cropped_face_t, w=0.5, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            if use_gpu:
                torch.cuda.empty_cache()
        except RuntimeError as error:
            print(f'Error: {error}')
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face)

    # 결과 생성
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    
    # 복원된 얼굴에 사각형 그리기
    if draw_rectangle:
        for bbox in face_helper.det_faces:
            x1, y1, x2, y2, _ = bbox.astype(int)
            cv2.rectangle(restored_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 최종 이미지 크기 조정 (원본 크기로)
    restored_img = cv2.resize(restored_img, (w, h), interpolation=cv2.INTER_LINEAR)

    return restored_img

class VideoFaceProcessor:
    def __init__(self, 
            base_image, 
            target_video, 
            tolerance=0.35, 
            output_video=None, 
            display_video=True, 
            display_rectangle=True, 
            segments=None,
            ctx_id=-1,
            ):

        # Initialize FaceAnalysis object
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=ctx_id)  # 0 Use GPU (set ctx_id=-1 to use CPU)

        # Load the reference face image and extract embedding
        ref_img = cv2.imread(base_image)
        if ref_img is None:
            raise FileNotFoundError(f"Unable to load the reference image: {base_image}")

        ref_faces = self.app.get(ref_img)
        if len(ref_faces) == 0:
            raise ValueError("No faces detected in the reference image.")

        # Extract embedding of the reference face (use the first face)
        self.known_face_embedding = ref_faces[0].embedding
        self.target_video = target_video
        self.output_video = output_video
        self.display_video = display_video
        self.display_rectangle = display_rectangle
        self.tolerance = tolerance
        self.specific_person_present = False  # Flag to indicate if Specific Person is present

        # Segments to process
        self.segments = self._prepare_segments(segments)
        
        # If output_video is None, do not use video saving feature
        self.fourcc = self._get_video_codec(output_video)

        self.trackers = []
        self.face_names = []
        self.face_similarities = []

    def _get_video_codec(self, output_video):
        if output_video is None:
            return None
        _, ext = os.path.splitext(output_video.lower())
        return cv2.VideoWriter_fourcc(*'VP90') if ext == '.webm' else cv2.VideoWriter_fourcc(*'mp4v')
        
    def _convert_to_frame_range(self, start_time_str, duration, fps):
        start_seconds = self._time_str_to_seconds(start_time_str)
        end_seconds = start_seconds + duration
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps)
        return start_frame, end_frame
        
    def _prepare_segments(self, segments):
        if segments is None:
            return None
        segment_frames = []
        fps = self.get_video_fps()
        for start_time_str, duration in segments:
            start_frame, end_frame = self._convert_to_frame_range(start_time_str, duration, fps)
            segment_frames.append((start_frame, end_frame))
        return segment_frames

    def _get_video_properties(self, video_capture):
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, width, height, total_frames

    def _initialize_video_writer(self, fps, width, height):
        if self.output_video:
            return cv2.VideoWriter(self.output_video, self.fourcc, fps, (width, height))
        return None

    def _calculate_similarities(self, faces):
        return [cosine_similarity([self.known_face_embedding], [face.embedding])[0][0] for face in faces]

    def _create_tracker(self, frame, bbox):
        tracker = cv2.legacy.TrackerKCF_create()
        x1, y1, x2, y2 = bbox
        tracker_bbox = (x1, y1, x2 - x1, y2 - y1)
        tracker.init(frame, tracker_bbox)
        return tracker

    def video_swap(self, func):
        def wrapper(*args, **kwargs):
            # Video capture
            video_capture = cv2.VideoCapture(self.target_video)

            # Get video properties
            fps, width, height, total_frames = self._get_video_properties(video_capture)

            video_writer = self._initialize_video_writer(fps, width, height)

            # Initialize variables
            self.trackers = []
            self.face_names = []
            self.face_similarities = []
            frame_skip = 24
            frame_count = 0

            # Convert segments to list of (start_frame, end_frame)
            segment_frames = self.segments if self.segments else [(0, total_frames)]

            # Process each segment
            for start_frame, end_frame in segment_frames:

                if start_frame >= total_frames:
                    print(f"Start frame {start_frame} exceeds total frames {total_frames}. Skipping segment.")
                    break

                # Adjust end_frame if it exceeds total_frames
                if end_frame > total_frames:
                    end_frame = total_frames

                # Set video capture to the start frame
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_count = start_frame

                while frame_count < end_frame:
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    frame_count += 1

                    if self.specific_person_present and frame_count % frame_skip > 0:
                        # Update trackers
                        self._update_trackers(frame, func)
                    else:
                        # Create new tracers
                        if self._process_first_tracker(frame, func) < 0:
                            continue

                    # Display current frame number / total frames at the top-left corner
                    cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Output or display video
                    if self.display_video:
                        cv2.imshow('Video', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    if video_writer:
                        video_writer.write(frame)

            # Cleanup
            video_capture.release()
            if video_writer:
                video_writer.release()
            if self.display_video:
                cv2.destroyAllWindows()

        return wrapper

    def get_video_fps(self):
        video_capture = cv2.VideoCapture(self.target_video)
        return video_capture.get(cv2.CAP_PROP_FPS)

    def _update_trackers(self, frame, func):

        new_trackers = []
        new_face_names = []
        new_face_similarities = []

        for tracker, name, similarity in zip(self.trackers, self.face_names, self.face_similarities):
            success, tracker_bbox = tracker.update(frame)
            if success:
                tracker_bbox = tuple(map(int, tracker_bbox))
                new_trackers.append(tracker)
                new_face_names.append(name)
                new_face_similarities.append(similarity)

                # Annotate frame and apply face swap if needed
                self._annotate_frame(tracker_bbox, frame, name, similarity, func)

            elif name == "Specific Person":
                self.specific_person_present = False

        # Update trackers and face info
        self.trackers = new_trackers
        self.face_names = new_face_names
        self.face_similarities = new_face_similarities

    def _process_first_tracker(self, frame, func)-> int:

        # Attempt to detect Specific Person in every frame
        self.trackers = []
        self.face_names = []
        self.face_similarities = []

        # Detect faces and extract embeddings
        faces = self.app.get(frame)                        
        if not faces:
            return -1  # No faces detected, skip to next frame

        similarities = self._calculate_similarities(faces)
        max_similarity = max(similarities, default=0)

        # Determine if the Specific Person is detected
        specific_person_index = similarities.index(max_similarity) if max_similarity > self.tolerance else None
        self.specific_person_present = specific_person_index is not None

        if self.specific_person_present:
            # Initialize trackers and annotate frame for Specific Person
            self._initialize_trackers(faces, frame, similarities, specific_person_index, func)
        else:
            
            for idx_face, face in enumerate(faces):
                bbox = face.bbox.astype(int)

                name = "Candidate" if similarities[idx_face] > self.tolerance else "Unknown"

                # Since we are not tracking, we do not initialize trackers
                # Annotate frame without applying face swap
                x1, y1, x2, y2 = bbox
                tracker_bbox = (x1, y1, x2 - x1, y2 - y1)
                                
                # Pass func=None to indicate no face swap should be applied
                self._annotate_frame(tracker_bbox, frame, name, similarities[idx_face], func=None)

        return 1

    def _initialize_trackers(self, faces, frame, similarities, specific_person_index, func):

        for idx, face in enumerate(faces):

            bbox = face.bbox.astype(int)

            if similarities[idx] > self.tolerance:                
                name = "Specific Person" if idx == specific_person_index else "Candidate"
            else:
                name = "Unknown"

            # Initialize tracker
            tracker = self._create_tracker(frame, bbox)
                                
            self.trackers.append(tracker)
            self.face_names.append(name)
            self.face_similarities.append(similarities[idx])

            # Annotate frame and apply face swap if needed
            x1, y1, x2, y2 = bbox
            tracker_bbox = (x1, y1, x2 - x1, y2 - y1)
            
            self._annotate_frame(tracker_bbox, frame, name, similarities[idx], func)

    def _annotate_frame(self, bbox, frame, name, similarity, func):

        left, top, width, height = map(int, bbox)
        expand_ratio = 0.3
        expand_width = int(width * expand_ratio)
        expand_height = int(height * expand_ratio)

        expanded_left = max(0, left - expand_width)
        expanded_top = max(0, top - expand_height)

        frame_height, frame_width, _ = frame.shape
        expanded_right = min(frame_width, left + width + expand_width)
        expanded_bottom = min(frame_height, top + height + expand_height)

        # Extract face region
        face_region = frame[expanded_top:expanded_bottom, expanded_left:expanded_right]

        if name == "Specific Person" and func:
            # Apply face swap
            swap_image = func(face_region)
            # Replace the face region with the swapped image
            if swap_image is not None:
                resized_swap_image = cv2.resize(swap_image, (expanded_right - expanded_left, expanded_bottom - expanded_top))
                frame[expanded_top:expanded_bottom, expanded_left:expanded_right] = resized_swap_image
            color = (0, 0, 255)  # Red
        elif name == "Candidate":
            color = (255, 0, 0)  # Blue
        else:
            color = (0, 255, 0)  # Green

        if self.display_rectangle:
            # Draw rectangle and annotations
            cv2.rectangle(frame, (expanded_left, expanded_top), (expanded_right, expanded_bottom), color, 2)
            cv2.putText(frame, name, (expanded_left, expanded_bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Similarity: {similarity:.2f}", (expanded_left, expanded_bottom + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _time_str_to_seconds(self, time_str):
        # Convert "mm:ss" format to total seconds
        minutes, seconds = map(int, time_str.split(':'))
        total_seconds = minutes * 60 + seconds
        return total_seconds

#Upscaling 모델 경로 설정
UPSCALE_MODEL_X2 = r"C:\pypjt\restore\Lib\site-packages\Real-ESRGAN\weights\RealESRGAN_x2plus.pth"
UPSCALE_MODEL_X4 = r"C:\pypjt\restore\Lib\site-packages\Real-ESRGAN\weights\RealESRGAN_x4plus.pth"

def upscale_image(input_image: np.ndarray, scale: Literal[2, 4] = 2) -> np.ndarray:
    """
    이미지를 업스케일링하는 함수.
    
    :param input_image: ndarray 타입의 입력 이미지
    :param model_path: 모델 파일 경로
    :param scale: 업스케일 배율 (기본값: 2)
    :return: 업스케일링된 ndarray 타입의 이미지
    """
    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Upscaing Model 선택
    model_path = UPSCALE_MODEL_X2 if scale == 2 else UPSCALE_MODEL_X4

    # 모델 생성
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    
    # 업스케일러 생성 (half=False로 설정하여 FP32 사용)
    upscaler = RealESRGANer(scale=scale, model_path=model_path, model=model, tile=400, tile_pad=10, pre_pad=0, half=False)
    
    # 입력 이미지가 BGR 형식일 경우, RGB로 변환
    if input_image.shape[-1] == 3:  # 이미지가 컬러일 경우
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # 업스케일링 수행
    output, _ = upscaler.enhance(input_image, outscale=scale)
    
    # 결과를 BGR로 다시 변환하여 반환
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    return output

def get_yaw_pitch_roll(face: Face) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    return face.pose[1], face.pose[0], face.pose[2] #yaw, pitch, roll