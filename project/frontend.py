import streamlit as st

st.title('Golf')

col1, col2 = st.columns(2)
with col1:
    st.header('src')
    video = open('teamproject_golf/data_folder/src/good.mp4', 'rb')
    video_bytes = video.read()
    st.video(video_bytes)
    
with col2:
    st.header('Landmarked')
    video = open('teamproject_golf/data_folder/outputs/output.mp4', 'rb')
    video_bytes = video.read()
    st.video(video_bytes)