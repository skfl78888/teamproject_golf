import os
import cv2
import streamlit as st
from inference.infer import PoseDetector, ActionClassifier

def run(src_video, params):
    global pose
    global ac
    src_dir = os.path.join('data_folder\src', src_video)
    actions = params.keys()
    means = {}
    for action in actions:
        means[action] = []
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
            means[action].append(ac.mean)
    return ac.esti_inform, means
            


pose, ac = PoseDetector(), ActionClassifier()
st.title('골프 AI 코치 학습 페이지')
st.subheader('처음 단계: 영상 선택 및 파라미터 조정')
src_video = st.selectbox('분석할 영상을 선택하여 주세요!', os.listdir('.\data_folder\src'))
st.write(f'[{src_video}] 을 선택하셨어요!')
if src_video:
    st.image(image=cv2.imread('utils_folder\pose_landmarks_reference.jpg'))
    st.write('자세별 파라미터 선택')
    tab1, tab2, tab3, tab4, tab5  = st.tabs(['address', 'backswing', 'top', 'impact', 'follow'])
    params = {}
    with tab1:
        k1 =st.multiselect('address', pose.landmarks_reference.keys())
        if k1:
            params['address'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k1}   
    with tab2:
        k2 = st.multiselect('backswing',  pose.landmarks_reference.keys())
        if k2:
            params['backswing'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k2} 
    with tab3:
        k3 =st.multiselect('top',  pose.landmarks_reference.keys())
        if k3:
            params['top'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k3}
    with tab4:
        k4 =st.multiselect('impact',  pose.landmarks_reference.keys())
        if k4:
            params['impact'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k4}
    with tab5:
        k5 =st.multiselect('follow',  pose.landmarks_reference.keys())
        if k5:
            params['follow'] = {land : st.slider(label=f'{land}', min_value=0.1, max_value=1.0, step=0.1, value=0.5) for land in k5}
    set_comp = st.button('다 했구, 분석 시작할게!')

if set_comp:
    estimation_informs, means = run(src_video, params)
    st.subheader('두번째 단계: 분석 결과')
    tabs = st.tabs(list(estimation_informs.keys()))
    for idx, action in enumerate(estimation_informs):
        with tabs[idx]:
            st.write('추정 frame:', estimation_informs[action]['frame'])
            st.write('Image')
            col1, col2 = st.columns(2)
            with col1:
                st.image(image=estimation_informs[action]['image'])
            with col2:
                dir_ = 'data_folder/labels/label_images/' + action + '.jpg'
                st.image(image=cv2.cvtColor(cv2.imread(dir_), cv2.COLOR_BGR2RGB))
            st.write('닮음 정도')
            st.area_chart(means[action])
            st.write('추정 좌표들')
            st.table(estimation_informs[action]['coordinate'])

    
    

    
        