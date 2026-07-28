import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm.wav")
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20

counter = 0
alarm_on = False

def calculate_EAR(eye_points, landmarks):
    points = []
    for point in eye_points:
        x = landmarks[point].x
        y = landmarks[point].y
        points.append([x, y])
        
    A = np.linalg.norm(
        np.array(points[1]) - np.array(points[5]))

    B = np.linalg.norm(
        np.array(points[2]) - np.array(points[4]))

    C = np.linalg.norm(
        np.array(points[0]) - np.array(points[3]))

    EAR = (A + B) / (2.0 * C)
    return EAR
    
cap = cv2.VideoCapture(0)
window_name = "Driver Drowsiness Detection"
cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_EAR = calculate_EAR(LEFT_EYE,landmarks)

            right_EAR = calculate_EAR(RIGHT_EYE,landmarks)
            EAR = (left_EAR + right_EAR) / 2

            cv2.putText(
                frame,
                f"EAR: {EAR:.2f}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)

            if EAR < EAR_THRESHOLD:
                counter += 1
                cv2.putText(frame,"DROWSINESS DETECTED!",(50, 100),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255),3)

                if counter >= FRAME_THRESHOLD:
                    if not alarm_on:
                        alarm.play()
                        alarm_on = True
            else:
                counter = 0
                if alarm_on:
                    alarm.stop()
                    alarm_on = False
    cv2.imshow(window_name,frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE) < 1:
        break

if alarm_on:
    alarm.stop()

cap.release()
cv2.destroyAllWindows()
pygame.quit()
