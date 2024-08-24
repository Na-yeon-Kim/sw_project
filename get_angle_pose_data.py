import cv2
import mediapipe as mp
import numpy as np
import csv

# 미디어파이프 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
collecting = False  # 데이터 수집 상태를 추적
data_buffer = []  # 수집된 데이터를 임시로 저장
label = None  # 입력받은 정수를 저장

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        result = pose.process(img_rgb)

        if result.pose_landmarks is not None:
            joint = np.zeros((33, 3))
            for j, lm in enumerate(result.pose_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # 각도를 계산하기 위해 연결할 관절 인덱스
            v1 = joint[[12, 12, 11, 11, 14, 13, 16, 15, 16, 15, 16, 15, 12, 11], :]
            v2 = joint[[14, 11, 12, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23], :]

            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 2, 1, 1, 0, 3, 4, 5, 4, 5, 10, 11], :],
                                        v[[1, 3, 12, 13, 4, 5, 6, 7, 10, 11, 8, 9], :]))
            angle = np.degrees(angle)

            if collecting:
                data = [label] + angle.tolist()
                data_buffer.append(data)

            # 포즈 랜드마크를 화면에 그리기
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pose Gesture', img)

    key = cv2.waitKey(1)

    if key == ord('s'):  # 데이터 수집 시작
        label = int(input("Enter a label for this pose set: "))
        data_buffer = []
        collecting = True
        print("Data collection started...")

    elif key == ord('f'):  # 데이터 수집 종료
        if collecting:
            collecting = False
            print("Data collection stopped.")

            # CSV 파일로 저장
            with open('pose_data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_buffer)
            print("Data saved to pose_data.csv.")

    elif key == 27:  # 'ESC' 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
