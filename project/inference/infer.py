from typing import Dict, List
import cv2
import numpy as np
import mediapipe as mp
import os

from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions import drawing_styles
from mediapipe.python.solutions import pose
# '''
# inference.py: loss 및 AI관련 파이썬 파일
# 작성담당자: 이원희박사님, 강경만과장님
# 해당 파이썬 구조는 얼마든지 구조변경 가능합니다!
# ※ pose_detector class는 신충호 사원이 작성할 예정.
# '''

class inference:
    def __init__(self):
        pass
    
    def loss(self):
        pass
        return None
    
    def inference(self):
        pass
        return None


class Networks:
    def __init__(self):
        pass
    
    def model(self):
        pass
        return None
    
    def __call__(self):
        pass
        return None


class PoseDetector:
    '''Google 연구팀에서 제공한 landmark 탐지 라이브러리를 이용하여 비디오의 landmark를 감지하여
    각 프레임에서의 원하는 landmark의 좌표를 제공하는 class입니다.
    
    Args(init timing이 아닌 call timing에서의 input 변수)
    - pose_prameters(dict 자료형): mediapipe 라이브러리 전용 hyper parameters
            'STATIC_IMAGE_MODE': #False: 감지 편하도록 video stream 임의로 조정 / True video stream 조정하지 않음 
            'MODEL_COMPLEXITY': #포즈 랜드마크 모델의 복잡성: 0, 1 또는 2. 랜드마크 정확도와 추론 지연은 일반적으로 모델 복잡성에 따라 올라갑니다.
            'SMOOTH_LANDMARKS': #True시 솔루션 필터가 여러 입력 이미지에 랜드마크를 표시하여 지터를 줄임. static_image_mode을 true로 설정하면 무시
            'ENABLE_SEGMENTATION': 현재 프로젝트에서 사용 X #Not used
            'SMOOTH_SEGMENTATION': 현재 프로젝트에서 사용 X  #Not used
            'MIN_DETECTION_CONFIDENCE': #탐지에 성공한 것으로 간주되는 사람 탐지 모델의 최소 신뢰 값([0.0, 1.0])입니다. 기본값은 0.5입니다.
            'MIN_TRACKING_CONFIDENCE': #포즈 랜드마크가 성공적으로 추적된 것으로 간주될 랜드마크 추적 모델의 최소 신뢰도 값, 그렇지 않으면 다음 입력 이미지에서 사람 감지가 자동으로 호출됩니다. 더 높은 값으로 설정하면 더 긴 대기 시간을 희생하면서 솔루션의 견고성을 높일 수 있습니다.
    - landmarks_name(list 자료형): 좌표를 받기위한 landmark모음
            종류: ['nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',]
        
    funtion definition: 좌표 제공 및 landmarked frame visualization
    - class 내 속성중 structure사용
        self.sturcture(Dict of np.ndarray)
        예) {'nose' : [[#1_frame x, #1_frame y, #1_frame z, #1_frame visibility],
                        ...
                        [#n_frame x, #n_frame y, #n_frame z, #n_frame visibility]]}
    
    '''
    def __init__(self, params, landmarks):
        self.mp_drawing = drawing_utils
        self.mp_drawing_styles = drawing_styles
        self.mp_pose = pose
        self.process = self.mp_pose.Pose(
            static_image_mode=params['STATIC_IMAGE_MODE'],
            model_complexity=params['MODEL_COMPLEXITY'],
            smooth_landmarks=params['SMOOTH_LANDMARKS'],
            enable_segmentation=params['ENABLE_SEGMENTATION'],
            smooth_segmentation=params['SMOOTH_SEGMENTATION'],
            min_detection_confidence=params['MIN_DETECTION_CONFIDENCE'],
            min_tracking_confidence=params['MIN_TRACKING_CONFIDENCE']
        )
        self.landmarks_reference = {
            'nose' : 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7, 
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10, 
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15, 
            'right_wrist': 16,
            'left_pinky': 17, 
            'right_pinky':18,
            'left_index': 19, 
            'right_index': 20,
            'left_thumb': 21, 
            'right_thumb':22,
            'left_hip': 23, 
            'right_hip':24,
            'left_knee': 25, 
            'right_knee':26,
            'left_ankle': 27, 
            'right_ankle': 28,
            'left_heel': 29, 
            'right_heel': 30,
            'left_foot_index': 31, 
            'right_foot_index': 32
        }
        self.structure = {}
        self.landmarks = landmarks
    
    def __call__(self, src_dir):
        self.making_landmarks_structure(landmarks=self.landmarks)
        video = cv2.VideoCapture(src_dir)
        fps = video.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./teamproject_golf/data_folder/outputs/output.avi', fourcc, fps, (640, 360))
        # i = 1
        # while video.isOpened():
        #     read_ok, frame = video.read()
        #     if not read_ok:
        #         print('TRK complete')
        #         break
        #     frame.flags.writeable = False
        #     res_obj = self.trk_process(image=frame)
        #     self.get_coordis(trk_obj=res_obj)
        #     landmarked_img = self.drawing_with_pose(frame, trk_obj=res_obj)
        #     out.write(landmarked_img)
        #     buf = './teamproject_golf/data_folder/outputs/res_imgs/' + str(i) + '.png'
        #     cv2.imwrite(buf, img=landmarked_img)
        #     # print(type(frame))
        #     # cv2.imshow('ggg', frame)
        #     # cv2.waitKey(1)
        #     i += 1
        
        img_path = "./teamproject_golf/data_folder/outputs/res_imgs/"
        
        frame_array = []
        for imgs in os.listdir(img_path):
            img = cv2.imread(os.path.join(img_path, imgs))
            frame_array.append(img)
            
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()
        
        for landmark in self.landmarks:
            self.structure[landmark] = np.delete(self.structure[landmark], [0, 0], 0)
    
    
    def making_landmarks_structure(self, landmarks: List):
        '''프로세스 중 원하는 landmark들을 이용해 Dict of list Data 구조를 생성합니다
        
        returns: dict of np.array
        exam {'picked landmark_1': [[x, y, z, visibility]] nd.array by frame}
        '''
        self.structure = {landmark: np.empty(shape=(1, 4)) for landmark in landmarks}
    
    def trk_process(self, image):
        '''이미지를 mediapipe라는 패키지를 사용하여 결과 객체를 반환합니다'''
        return self.process.process(image=image)       
    
    def get_coordis(self, trk_obj):
        '''이미지에서 연산한 landmark들의 좌표를 structure에 append합니다
        (이미지 1개에 한함, return X)'''
        keys = list(self.structure.keys())
        for key in keys:
            idx = self.landmarks_reference[key]
            self.structure[key] = np.append(arr=self.structure[key], 
                                    values=[[trk_obj.pose_landmarks.landmark[idx].x, 
                                    trk_obj.pose_landmarks.landmark[idx].y,
                                    trk_obj.pose_landmarks.landmark[idx].z,
                                    trk_obj.pose_landmarks.landmark[idx].visibility]], 
                    axis=0)
    
    def drawing_with_pose(self, image, trk_obj):
        '''원본 이미지를 복사하여 landmark들이 추가된 이미지를 만들어 반환합니다'''
        copied_img = image.copy()
        copied_img.flags.writeable = True
        self.mp_drawing.draw_landmarks(image=copied_img, 
                                    landmark_list= trk_obj.pose_landmarks,
                                    connections=self.mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        landmarked_image = cv2.resize(copied_img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        return landmarked_image 