# ''' 
# main.py: 모듈(class 모음)
# pipelines 설계
# io처리 설계 
# 작성담당자: 신충호사원
# '''
from multiprocessing import Process
from inference.infer import PoseDetector
import os



pose_params = {
        'STATIC_IMAGE_MODE': False, #False: 감지 편하도록 video stream 임의로 조정 / True video stream 조정하지 않음 
        'MODEL_COMPLEXITY': 1, #포즈 랜드마크 모델의 복잡성: 0, 1 또는 2. 랜드마크 정확도와 추론 지연은 일반적으로 모델 복잡성에 따라 올라갑니다.
        'SMOOTH_LANDMARKS': True, #True시 솔루션 필터가 여러 입력 이미지에 랜드마크를 표시하여 지터를 줄임. static_image_mode을 true로 설정하면 무시
        'ENABLE_SEGMENTATION': False, #Not used
        'SMOOTH_SEGMENTATION': True, #Not used
        'MIN_DETECTION_CONFIDENCE': 0.5, #탐지에 성공한 것으로 간주되는 사람 탐지 모델의 최소 신뢰 값([0.0, 1.0])입니다. 기본값은 0.5입니다.
        'MIN_TRACKING_CONFIDENCE': 0.5 #포즈 랜드마크가 성공적으로 추적된 것으로 간주될 랜드마크 추적 모델의 최소 신뢰도 값, 그렇지 않으면 다음 입력 이미지에서 사람 감지가 자동으로 호출됩니다. 더 높은 값으로 설정하면 더 긴 대기 시간을 희생하면서 솔루션의 견고성을 높일 수 있습니다.
    }
landmarks = ['nose', 'left_elbow']
dir = 'teamproject_golf/data_folder/src/good.mp4' 


if __name__ == "__main__":
    pose = PoseDetector(pose_params, landmarks)
    pose(dir)
    
    