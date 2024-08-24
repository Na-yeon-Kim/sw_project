import cv2
import mediapipe as mp
import numpy as np
import joblib  # joblib으로 모델 로드

# 미디어파이프 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 학습된 kNN 모델 로드
model_filename = r'knn_model_pose_97.pkl'
knn = joblib.load(model_filename)

cap = cv2.VideoCapture(0)

# Pose 객체를 루프 밖에서 생성
with mp_pose.Pose(static_image_mode=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = pose.process(img_rgb)

        if result.pose_landmarks is not None:
            joint = np.zeros((33, 3))
            for j, lm in enumerate(result.pose_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            v1 = joint[[12, 12, 11, 11, 14, 13, 16, 15, 16, 15, 16, 15, 12, 11], :]
            v2 = joint[[14, 11, 12, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 2, 1, 1, 0, 3, 4, 5, 4, 5, 10, 11], :],
                                        v[[1, 3, 12, 13, 4, 5, 6, 7, 10, 11, 8, 9], :]))
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

            # 포즈 랜드마크를 화면에 그리기
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose Recognition', img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 키를 누르면 종료
            break

cap.release()
cv2.destroyAllWindows()