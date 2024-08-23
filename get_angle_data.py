import cv2
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 제스처 정의 (가위, 바위, 보 등)
rps_gesture = {0: 'rock', 1: 'paper', 2: 'scissors'}

cap = cv2.VideoCapture(0)
collecting = False  # 데이터 수집 상태를 추적
data_buffer = []  # 수집된 데이터를 임시로 저장
label = None  # 입력받은 정수를 저장

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5) as hands:

        result = hands.process(img)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]#벡터 시점 설정
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]#벡터 종점 설정
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',#벡터 사이 각 도출
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)

                if collecting:
                    data = [label] + angle.tolist()
                    data_buffer.append(data)

                # 제스처 인식 부분 (옵션)
                # data = np.array([angle], dtype=np.float32)
                # ret, results, neighbours, dist = knn.findNearest(data, 3)
                # idx = int(results[0][0])

                # if idx in rps_gesture.keys():
                #     cv2.putText(img, text=rps_gesture[idx].upper(),
                #                 org=(int(res.landmark[0].x * img.shape[1]),
                #                      int(res.landmark[0].y * img.shape[0] + 20)),
                #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #                 fontScale=1, color=(255, 255, 255), thickness=2)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Gesture', img)

    key = cv2.waitKey(1)

    if key == ord('s'):  # 데이터 수집 시작
        label = int(input("Enter a label for this gesture set: "))
        data_buffer = []
        collecting = True
        print("Data collection started...")

    elif key == ord('f'):  # 데이터 수집 종료
        if collecting:
            collecting = False
            print("Data collection stopped.")

            # CSV 파일로 저장
            with open('gesture_data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(data_buffer)
            print("Data saved to gesture_data.csv.")

    elif key == 27:  # 'ESC' 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
