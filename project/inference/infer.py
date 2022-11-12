from typing import Dict, List
import cv2
import numpy as np
import mediapipe as mp
import os
import json

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


class ActionClassifier:
    def __init__(self):
        self.label_axises = self.get_json('data_folder\labels\jsons')
        
        self.esti_inform = {}
        
    def get_json(self, json_dir):
        result = {}
        actions_json = os.listdir(json_dir)
        for action_json in actions_json:
            dir =  os.path.join(json_dir, action_json)
            action = action_json.split('.')[0]
            with open(dir) as f:
                axis = json.load(f)
            result[action] = axis        
        return result
    
    def update_information(self, action: str, parameter: dict, pose_coordis: dict, num_frame: int, image:np.ndarray):
        similarities = [self.calculator(self.label_axises[action][landmark], pose_coordis[landmark]) for landmark in parameter[action].keys()]
        similar_mean = np.dot(np.array(similarities), np.array(list(parameter[action].values()))) / len(parameter[action])
        self.a = similar_mean
        if not action in list(self.esti_inform.keys()):
            self.esti_inform[action] = {'frame': num_frame, 'similar': similar_mean, 'image': image}
        else:
            if self.esti_inform[action]['similar'] <= similar_mean: 
                self.esti_inform[action]['frame'] = num_frame
                self.esti_inform[action]['similar'] =  similar_mean 
                self.esti_inform[action]['image'] =  image
                
    def calculator(self, label_vec: list, pose_vec: list):
        '''벡터의 코사인 유사도 계산
        formula = 내적(a,b) / (a의 2norm * b의 2norm)'''
        pose_np = np.array(pose_vec)
        label_np = np.array(label_vec)
        return (pose_np @ label_np) / (np.linalg.norm(pose_np, 2) * np.linalg.norm(label_np, 2))
    
    def restet_inform(self):
        self.esti_inform = {'address': {},
                            'backswing': {},
                            'top':{},
                            'follow': {},
                            'impact': {}}
        
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
        'left_pinky', 'right_pinky',
        'left_index', 'right_index',
        'left_thumb', 'right_thumb',
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
    def __init__(self):
        self.mp_drawing = drawing_utils
        self.mp_drawing_styles = drawing_styles
        self.mp_pose = pose
        self.params = {
        'STATIC_IMAGE_MODE': False, #False: 감지 편하도록 video stream 임의로 조정 / True video stream 조정하지 않음 
        'MODEL_COMPLEXITY': 1, #포즈 랜드마크 모델의 복잡성: 0, 1 또는 2. 랜드마크 정확도와 추론 지연은 일반적으로 모델 복잡성에 따라 올라갑니다.
        'SMOOTH_LANDMARKS': True, #True시 솔루션 필터가 여러 입력 이미지에 랜드마크를 표시하여 지터를 줄임. static_image_mode을 true로 설정하면 무시
        'ENABLE_SEGMENTATION': False, #Not used
        'SMOOTH_SEGMENTATION': True, #Not used
        'MIN_DETECTION_CONFIDENCE': 0.5, #탐지에 성공한 것으로 간주되는 사람 탐지 모델의 최소 신뢰 값([0.0, 1.0])입니다. 기본값은 0.5입니다.
        'MIN_TRACKING_CONFIDENCE': 0.5 #포즈 랜드마크가 성공적으로 추적된 것으로 간주될 랜드마크 추적 모델의 최소 신뢰도 값, 그렇지 않으면 다음 입력 이미지에서 사람 감지가 자동으로 호출됩니다. 더 높은 값으로 설정하면 더 긴 대기 시간을 희생하면서 솔루션의 견고성을 높일 수 있습니다.
        }
        self.process = self.mp_pose.Pose(
            static_image_mode=self.params['STATIC_IMAGE_MODE'],
            model_complexity=self.params['MODEL_COMPLEXITY'],
            smooth_landmarks=self.params['SMOOTH_LANDMARKS'],
            enable_segmentation=self.params['ENABLE_SEGMENTATION'],
            smooth_segmentation=self.params['SMOOTH_SEGMENTATION'],
            min_detection_confidence=self.params['MIN_DETECTION_CONFIDENCE'],
            min_tracking_confidence=self.params['MIN_TRACKING_CONFIDENCE']
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
        self.coordinates = {}
        self.landmarks = list(self.landmarks_reference.keys())

    # def __call__(self, src_dir):
    #     self.making_landmarks_structure(landmarks=self.landmarks)
    #     video = cv2.VideoCapture(src_dir)
    #     fourcc = cv2.VideoWriter_fourcc(*'h', '2', '6', '4')
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #     # out = cv2.VideoWriter('./teamproject_golf/data_folder/outputs/mac_land.mp4', fourcc, fps, size)
    #     print(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #     while video.isOpened():
    #         cnt = video.get(cv2.CAP_PROP_POS_FRAMES)
    #         read_ok, frame = video.read()
    #         if not read_ok:
    #             print('TRK complete')
    #             break
    #         frame.flags.writeable = False
    #         res_obj = self.trk_process(image=frame)
    #         self.get_coordis(trk_obj=res_obj)
            
    #         # cv2.imshow('ggg', frame)
    #         landmarked_img = self.drawing_with_pose(frame, trk_obj=res_obj)
    #         # out.write(landmarked_img)
    #         # height, weight, _ = landmarked_img.shape
    #         # size = (weight, height)
    #         # cv2.waitKey(0)
    #     video.release()
    #     # out.release()
        
    #     # for landmark in self.landmarks:
    #     #     self.coordinates[landmark] = np.delete(self.coordinates[landmark], [0, 0], 0)
    
    def trk_process(self, image):
        '''이미지를 mediapipe라는 패키지를 사용하여 결과 객체를 반환합니다'''
        return self.process.process(image=image)       
    
    def get_coordis_3d(self, trk_obj):
        '''이미지에서 연산한 landmark들의 좌표를 coordinate에 apdate합니다
        (이미지 1개에 한함, return X)'''
        for key in self.landmarks:
            idx = self.landmarks_reference[key]
            self.coordinates[key] = np.array( 
                                    [trk_obj.pose_world_landmarks.landmark[idx].x, 
                                    trk_obj.pose_world_landmarks.landmark[idx].y,
                                    trk_obj.pose_world_landmarks.landmark[idx].z,]
                                    )
    

    def drawing_with_pose(self, image, trk_obj):
        '''원본 이미지를 복사하여 landmark들이 추가된 이미지를 만들어 반환합니다'''
        copied_img = image.copy()
        copied_img.flags.writeable = True
        self.mp_drawing.draw_landmarks(image=copied_img, 
                                    landmark_list= trk_obj.pose_landmarks,
                                    connections=self.mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        return copied_img 
    
    '''pose추론하고 시간에 따른 룰베이스 조건으로 사진 및 landmarks 재생성'''
    '''백스윙 할 때만 무빙에버리지 할것'''