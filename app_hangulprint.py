import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from unicode import join_jamos

# 미디어파이프 설정
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 주어진 딕셔너리
char_dict = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ',
    8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ',
    15: 'ㅐ', 16: 'ㅑ', 17: 'ㅒ', 18: 'ㅓ', 19: 'ㅔ', 20: 'ㅕ', 21: 'ㅖ',
    22: 'ㅗ', 23: 'ㅛ', 24: 'ㅜ', 25: 'ㅠ', 26: 'ㅡ', 27: 'ㅣ'
}

# 학습된 kNN 모델 로드
model_filename = r'C:\sw\first_project\knn_model_96.pkl'
knn = joblib.load(model_filename)

cap = cv2.VideoCapture(0)

def draw_text_pil(image, text, position, font_size=30, color=(255, 255, 255)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("gulim.ttc", font_size)  # 변경: 적절한 한국어 폰트 파일로 수정 필요
    except IOError:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5) as hands:

    prediction_list = []
    input_mode = False  # 손 모양 입력 모드
    display_text = ""  # 화면에 표시할 텍스트
    show_hangul = False  # 한글 표시 여부

    start_time = time.time()

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

                if input_mode:
                    data = np.array([angle], dtype=np.float32)
                    prediction = knn.predict(data)
                    predicted_value = int(prediction[0])

                    cv2.putText(img, text=f'Predicted Value: {predicted_value}',
                                org=(10, 50),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1, color=(255, 255, 255), thickness=2)

                    if not prediction_list or prediction_list[-1] != predicted_value:
                        prediction_list.append(predicted_value)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                current_time = time.time()

                if current_time - start_time >= 1:
                    start_time = current_time

        if input_mode:
            display_text = "Press F to stop input"
        else:
            display_text = "Press S to start input"

        if show_hangul:
            # 한글 문자열을 화면에 표시
            hangul_text = ''.join([char_dict[num] for num in prediction_list])
            img = draw_text_pil(img, join_jamos(hangul_text), (10, img.shape[0] - 50), font_size=30)

        img = draw_text_pil(img, display_text, (10, img.shape[0] - 10), font_size=20)

        cv2.rectangle(img, (img.shape[1]-100, img.shape[0]-100), (img.shape[1]-50, img.shape[0]-50), (0, 0, 255), -1)
        cv2.putText(img, 'Press R', (img.shape[1]-120, img.shape[0]-110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Hand Gesture Recognition', img)

        key = cv2.waitKey(1)
        if key == 27:  # ESC 키를 누르면 종료
            break
        elif key == ord('s'):  # 's' 키를 누르면 손 모양 입력 모드로 변경
            input_mode = True
        elif key == ord('f'):  # 'f' 키를 누르면 손 모양 입력 모드 해제
            input_mode = False
        elif key == ord('r'):  # 'r' 키를 누르면 마지막 예측값 추가
            if prediction_list:
                prediction_list.append(prediction_list[-1])
        elif key == ord('t'):  # 't' 키를 누르면 한글 문자열 표시
            show_hangul = not show_hangul

cap.release()
cv2.destroyAllWindows()

# 최종 예측 결과 리스트 출력
print("Predictions every second:", prediction_list)

result = ''.join([char_dict[num] for num in prediction_list])
print("Hangul:", result)
