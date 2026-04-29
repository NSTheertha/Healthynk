from imutils.video import VideoStream 
from imutils import face_utils 
import imutils 
import time 
import dlib 
import cv2 
import numpy as np 
from EAR import eye_aspect_ratio 
from MAR import mouth_aspect_ratio 
from HeadPose import getHeadTiltAndCoords 
import pygame 
pygame.init() 
pygame.mixer.music.load("alarm.wav") 
print("[INFO] loading facial landmark predictor...") 
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
print("[INFO] initializing camera...") 
vs = VideoStream(src=0).start() 
time.sleep(2.0) 
frame_width = 1024 
frame_height = 576 
image_points = np.array([ 
(359, 391), # Nose tip 
(399, 561), # Chin 
(337, 297), # Left eye 
(513, 301), # Right eye 
(345, 465), # Left mouth 
(453, 469) 
# Right mouth 
], dtype="double") 
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] 
(mStart, mEnd) = (49, 68) 
FATIGUE_EAR_THRESH = 0.25 
FATIGUE_MAR_THRESH = 0.79 
while True: 
frame = vs.read() 
frame = imutils.resize(frame, width=frame_width, height=frame_height) 
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
size = gray.shape 
rects = detector(gray, 0) 
if len(rects) > 0: 
cv2.putText(frame, f"{len(rects)} face(s) found", (10, 20), 
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
for rect in rects: 
(bX, bY, bW, bH) = face_utils.rect_to_bb(rect) 
cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1) 
shape = predictor(gray, rect) 
shape = face_utils.shape_to_np(shape) 
leftEye = shape[lStart:lEnd] 
rightEye = shape[rStart:rEnd] 
ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0 
eye_fatigue = 1 if ear < FATIGUE_EAR_THRESH else 0 
cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1) 
cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1) 
mouth = shape[mStart:mEnd] 
mar = mouth_aspect_ratio(mouth) 
mouth_fatigue = 1 if mar > FATIGUE_MAR_THRESH else 0 
cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1) 
for (i, (x, y)) in enumerate(shape): 
cv2.circle(frame, (x, y), 1, (0, 0, 255), -1) 
if i in [33, 8, 36, 45, 48, 54]: 
image_points[[33,8,36,45,48,54].index(i)] = np.array([x, y], 
dtype='double') 
(head_tilt_degree, start_point, end_point, 
end_point_alt) = getHeadTiltAndCoords(size, image_points, 
frame_height) 
cv2.line(frame, start_point, end_point, (255, 0, 0), 2) 
cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2) 
head_fatigue = 0 
if head_tilt_degree and abs(head_tilt_degree[0]) > 15: 
head_fatigue = 1 
FATIGUE_SCORE = eye_fatigue + mouth_fatigue + head_fatigue 
36 
if FATIGUE_SCORE == 0: 
fatigue_state = "FATIGUE: NORMAL" 
color = (0, 255, 0) 
elif FATIGUE_SCORE == 1: 
fatigue_state = "FATIGUE: MODERATE" 
color = (0, 255, 255) 
else: 
fatigue_state = "FATIGUE: HIGH" 
color = (0, 0, 255) 
try: 
pygame.mixer.music.play() 
except: 
pass 
cv2.putText(frame, fatigue_state, (450, 50), 
cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2) 
cv2.putText(frame, f"EAR: {ear:.2f}", (10, 80), 
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 
cv2.putText(frame, f"MAR: {mar:.2f}", (10, 100), 
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 
cv2.putText(frame, f"Head Tilt: {head_tilt_degree[0] if head_tilt_degree 
else 0}", 
(10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) 
cv2.imshow("Fatigue Monitoring System", frame) 
if cv2.waitKey(1) & 0xFF == ord("q"): 
break 
cv2.destroyAllWindows() 
vs.stop() 
