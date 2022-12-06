import os
import cv2
import json
import copy
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from inference.infer import PoseDetector, ActionClassifier
from inference.rules import Rules

def draw_color(x):
    if x == '우수':
        color = f'background-color: #94f48c'
        return color
    elif x == '보통':
        color = f'background-color: #f4eb9d'
        return color
    elif x == '교정필요':
        color = f'background-color: #ef8066'
        return color
    else:
        return ''

def run(src_video, params):
    global pose
    global ac
    src_dir = os.path.join('data_folder/src', src_video)
    actions = params.keys()
    for action in actions:
        video = cv2.VideoCapture(src_dir)
        while video.isOpened():
            read_ok, frame = video.read()
            if not read_ok:
                print(f'TRK Complete!:{action}')
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

pose, ac, rule = PoseDetector(), ActionClassifier(), Rules()
st.title('골프 AI 코치')
st.caption('Made by 강경만, 이원희, 박희용, 손명원, 신충호 \n 2022 하반기 포스코 AI 전문가 과정')
st.header('1단계 영상선택')
src_video = st.selectbox('분석할 영상을 선택하여 주세요!', os.listdir('data_folder/src'))
st.caption(f'[{src_video}] 을 선택하셨어요!')
set_comp = st.button('분석 시작')
for column, guide in zip(st.columns(2), ['Input', 'Reference']):
    with column:
        if guide == 'Input':
            video_dir = os.path.join('data_folder/src', src_video)
        else:
            video_dir = 'data_folder/labels/mac.mp4'
        st.subheader(guide)
        st.video(video_dir, 'rb')

if set_comp:
    with open('data_folder/parameter/params.json', 'r') as r:
        params = json.load(r)
    with st.spinner('분석 중 입니다...'):
        estimation_informs = run(src_video, params)
    st.success('Done!')
    st.header('2단계 분석결과')
    tabs = st.tabs(list(estimation_informs.keys()))
    with open('data_folder/parameter/label_rule_value.json', 'rb') as r:
        label_rule = json.load(r)
    esti_rule = rule.rule_structure
    for idx, action in enumerate(estimation_informs):
        with tabs[idx]:
            st.write('Image')
            col1, col2 = st.columns(2)
            with col1:
                st.image(image=estimation_informs[action]['image'])
            with col2:
                dir_ = 'data_folder/labels/label_images/' + action + '.jpg'
                st.image(image=cv2.cvtColor(cv2.imread(dir_), cv2.COLOR_BGR2RGB))
            esti_rule = rule.handle(action=action, 
                        value=esti_rule, 
                        coordis=estimation_informs[action]['coordinate'])
            data = {}
            for col in ['추정', '프로', '분석결과']:
                if col == '추정':
                    data[col] = esti_rule[action]  
                elif col == '프로': 
                    data[col] = {i : label_rule[action][i]['mean'] for i in label_rule[action].keys()}
                else:
                    data[col] = {i : rule.get_grade(esti_rule[action][i], label_rule[action][i]) for i in label_rule[action].keys()}
            
            df = pd.DataFrame(data=data)
            
            st.dataframe(df.style.applymap(draw_color))
            # st.write(label_rule[action])