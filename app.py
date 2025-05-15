import cv2
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoTransformerBase
)
from detector import detect

st.title("🚗 Real‑Time Driver Drowsiness Detection")

class Transformer(VideoTransformerBase):
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        status = detect(img)
        if status:
            color = (0,0,255) if status == "DROWSY" else (0,255,0)
            cv2.putText(
                img,
                status,
                (30,50),
                self.font,
                1.5,
                color,
                3,
                cv2.LINE_AA
            )
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

webrtc_streamer(
    key="drowsiness",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=Transformer,
    media_stream_constraints={"video": True, "audio": False}
)
