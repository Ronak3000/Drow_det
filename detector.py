import cv2
import numpy as np
import mediapipe as mp
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTilt

# Use streaming (not static) for speed:
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def detect(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]
    coords = np.array([[int(p.x*w), int(p.y*h)] for p in lm])

    LEFT  = [33,160,158,133,153,144]
    RIGHT = [362,385,387,263,373,380]
    MOUTH = [61,291,0,17,13,14,87,317]

    ear = (eye_aspect_ratio(coords[LEFT]) +
           eye_aspect_ratio(coords[RIGHT])) / 2
    mar = mouth_aspect_ratio(coords[MOUTH])

    # 6 points for head‐pose:
    pts = np.zeros((6,2), dtype="double")
    pts[0] = coords[1]
    pts[1] = coords[199]
    pts[2] = coords[LEFT[0]]
    pts[3] = coords[RIGHT[3]]
    pts[4] = coords[MOUTH[0]]
    pts[5] = coords[MOUTH[1]]
    tilt = getHeadTilt(frame.shape[:2], pts)

    status = "DROWSY" if (ear < 0.2 or mar > 0.79 or tilt > 15) else "ALERT"
    return status
