import cv2
import mediapipe as mp
import numpy as np
import joblib  # joblib으로 모델 로드

# 미디어파이프 설정
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 학습된 kNN 모델 로드
model_filename = r'C:\sw\first_project\knn_model_96.pkl'
knn = joblib.load(model_filename)

cap = cv2.VideoCapture(0)

# Hands 객체를 루프 밖에서 생성
with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5) as hands:

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = hands.process(img_rgb)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)

                # 예측 수행
                data = np.array([angle], dtype=np.float32)
                prediction = knn.predict(data)
                predicted_value = int(prediction[0])

                # 예측된 정수값을 화면에 표시
                cv2.putText(img, text=f'Predicted Value: {predicted_value}',
                            org=(10, 50),  # 화면에 표시할 위치
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 255, 255), thickness=2)

                # 손 랜드마크를 화면에 그리기
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Gesture Recognition', img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 키를 누르면 종료
            break

cap.release()
cv2.destroyAllWindows()