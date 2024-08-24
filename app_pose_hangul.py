import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import ImageFont, ImageDraw, Image

# 미디어파이프 설정
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 학습된 kNN 모델 로드
model_filename = r'knn_model_pose_97.pkl'
knn = joblib.load(model_filename)

# TTC 폰트 설정 (예: Apple SD Gothic Neo.ttc)
ttc_font_path = "gulim.ttc"
font = ImageFont.truetype(ttc_font_path, 40)  # 폰트 크기를 설정하세요

cap = cv2.VideoCapture(0)

# 반복되는 예측 결과를 추적하기 위한 변수 설정
previous_value = None
repetition_count = 0
max_repetitions = 3  # 반복 횟수 기준 (3번 반복되는 경우)

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

            # 반복되는 예측 결과를 감지
            if predicted_value == previous_value:
                repetition_count += 1
            else:
                repetition_count = 1  # 새로운 값이 나오면 카운트를 1로 설정
            previous_value = predicted_value

            # 이미지를 PIL로 변환
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)

            # 반복이 감지되면 "아프다" 또는 "덥다" 표시
            if repetition_count >= max_repetitions:
                if (previous_value == 11):
                    draw.text((10, 100), '아프다', font=font, fill=(0, 0, 255))
                elif (previous_value == 21):
                    draw.text((10, 100), '덥다', font=font, fill=(0, 0, 255))
            else:
                draw.text((10, 50), f'Predicted Value: {predicted_value}', font=font, fill=(255, 255, 255))

            # PIL 이미지를 다시 OpenCV 이미지로 변환
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 포즈 랜드마크를 화면에 그리기
            mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose Recognition', img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 키를 누르면 종료
            break

cap.release()
cv2.destroyAllWindows()
