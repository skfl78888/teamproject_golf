import os
import cv2
import streamlit as st
from inference.infer import PoseDetector, ActionClassifier

def run(src_video, params):
    pose, ac = PoseDetector(), ActionClassifier()
    src_dir = os.path.join('data_folder\src', src_video)
    actions = params.keys()
    for action in actions:
        video = cv2.VideoCapture(src_dir)
        while video.isOpened():
            read_ok, frame = video.read()
            if not read_ok:
                print('TRK Complete!')
                break
            cnt = video.get(cv2.CAP_PROP_POS_FRAMES)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, dsize=(0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
            pose_obj = pose.trk_process(image=frame)
            pose.get_coordis_3d(pose_obj)
            landmarked_img = pose.drawing_with_pose(frame, pose_obj)
            ac.update_information(action=action,
                                parameter=params,
                                pose_coordis=pose.coordinates,
                                num_frame=cnt, image=landmarked_img)
    return ac.esti_inform
            

st.title('골프 AI 코치 학습 페이지')
st.subheader('처음 단계: 영상 선택 및 파라미터 조정')
src_video = st.selectbox('분석할 영상을 선택하여 주세요!', os.listdir('.\data_folder\src'))
st.write(f'[{src_video}] 을 선택하셨어요!')
if src_video:
    st.image(image=cv2.imread('utils_folder\pose_landmarks_reference.jpg'))
    st.write('자세별 파라미터 선택')
    col1, col2, col3, col4, col5  = st.columns(5)
    params = {}
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
            params['address'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k1}   
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
            params['backswing'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k2} 
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
            params['top'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k3}
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
            params['impact'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k4}
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
            params['follow'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k5}
    set_comp = st.button('다 했구, 분석 시작할게!')

if set_comp:
    estimation_informs = run(src_video, params)
    st.subheader('두번째 단계: 분석 결과')
    print(estimation_informs)
    for action in estimation_informs:
        st.write(f'{action} 자세')
        col1, col2 = st.columns(2)
        with col1:
            st.image(image=estimation_informs[action]['image'])
        with col2:
            dir_ = 'data_folder/labels/label_images/' + action + '.jpg'
            st.image(image=cv2.cvtColor(cv2.imread(dir_), cv2.COLOR_BGR2RGB))
    
    
    
    

    
        