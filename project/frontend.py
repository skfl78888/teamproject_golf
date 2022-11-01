import streamlit as st

st.title('Golf')

st.header('video')
st.subheader('src')
video = open('teamproject_golf/data_folder/outputs/output.mp4', 'rb')
video_bytes = video.read()

st.video(video_bytes)