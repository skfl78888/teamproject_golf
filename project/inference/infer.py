from typing import Dict, List
import cv2
import mediapipe as mp
# import numpy as np


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
    
    def call(self):
        pass
        return None


class PoseDetector:
    def __init__(self):
        pass
    
    def set_parameters(self):
        pass
        return None
    
    def tracking_pose(self, dir, trk_points: List, params:Dict):
        video = cv2.VideoCapture(dir)
        pose = mp.solutions.pose.Pose(
            static_image_mode=params['STATIC_IMAGE_MODE'],
            model_complexity=params['MODEL_COMPLEXITY'],
            smooth_landmarks=params['SMOOTH_LANDMARKS'],
            enable_segmentation=params['ENABLE_SEGMENTATION'],
            smooth_segmentation=params['SMOOTH_SEGMENTATION'],
            min_detection_confidence=params['MIN_DETECTION_CONFIDENCE'],
            min_tracking_confidence=params['MIN_TRACKING_CONFIDENCE']
        )
        while video.isOpened():
            read_ok, image = video.read()
            image = cv2.flip(image, 0)
            if not read_ok:
                print('TRK Complete')
                break
            image.flags.writeable = False
            pose_res = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # print(pose_res.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x)
            randmarks = [pose_res.pose_landmarks.landmark[i] for i in range(2)]
            
            image.flags.writeable = True
            mp.solutions.drawing_utils.draw_landmarks(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                pose_res.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )
            cv2.imshow('MediaPipe Pose', image)
        # video.release()
        # cv2.destroyAllWindows()
        return pose_res
    
    def drawing_with_pose(self, dir, pose_obj):
        pass
    
    def call(self):
        pass
    
    def pipelines_buf(self):
        pose_params = {
            'STATIC_IMAGE_MODE': False, #False: 감지 편하도록 video stream 임의로 조정 / True video stream 조정하지 않음 
            'MODEL_COMPLEXITY': 1.0, #포즈 랜드마크 모델의 복잡성: 0, 1 또는 2. 랜드마크 정확도와 추론 지연은 일반적으로 모델 복잡성에 따라 올라갑니다.
            'SMOOTH_LANDMARKS': True, #True시 솔루션 필터가 여러 입력 이미지에 랜드마크를 표시하여 지터를 줄임. static_image_mode을 true로 설정하면 무시
            'ENABLE_SEGMENTATION': False, #Not used
            'SMOOTH_SEGMENTATION': True, #Not used
            'MIN_DETECTION_CONFIDENCE': 0.5, #탐지에 성공한 것으로 간주되는 사람 탐지 모델의 최소 신뢰 값([0.0, 1.0])입니다. 기본값은 0.5입니다.
            'MIN_TRACKING_CONFIDENCE': 0.5 #포즈 랜드마크가 성공적으로 추적된 것으로 간주될 랜드마크 추적 모델의 최소 신뢰도 값, 그렇지 않으면 다음 입력 이미지에서 사람 감지가 자동으로 호출됩니다. 더 높은 값으로 설정하면 더 긴 대기 시간을 희생하면서 솔루션의 견고성을 높일 수 있습니다.
        }