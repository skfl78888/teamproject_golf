import os
import cv2
import streamlit as st

st.title('골프 AI 코치 학습 페이지')
st.subheader('처음 단계: 영상 선택 및 파라미터 조정')
option = st.selectbox('분석할 영상을 선택하여 주세요!', os.listdir('.\data_folder\src'))
st.write(f'[{option}] 을 선택하셨어요!')
if option:
    st.image(image=cv2.imread('utils_folder/pose_landmarks_reference.jpg'))
    st.write('자세별 파라미터 선택')
    col1, col2, col3, col4, col5  = st.columns(5)
    with col1:
        k1 =st.multiselect('address',  ['nose',
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
        'left_foot_index', 'right_foot_index',])
        if k1:
            address_weights = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k1}   
    with col2:
        k2 = st.multiselect('backswing',  ['nose',
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
        'left_foot_index', 'right_foot_index',])
        if k2:
            backswing_weights = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k2} 
    with col3:
        k3 =st.multiselect('top',  ['nose',
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
        'left_foot_index', 'right_foot_index',])
        if k3:
            top_weights = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k3}
    with col4:
        k4 =st.multiselect('impact',  ['nose',
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
        'left_foot_index', 'right_foot_index',])
        if k4:
            impact_weights = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k4}
    with col5:
        k5 =st.multiselect('follow',  ['nose',
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
        'left_foot_index', 'right_foot_index',])
        if k5:
            follow_weights = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k5}
    set_comp = st.button('다 했구, 분석 시작할게!')

if set_comp:
    st.subheader('두번째 단계: 분석 결과')