import os
import cv2
import mediapipe as mp
import csv
from datetime import datetime

# 손 랜드마크를 감지하고 그리기 위한 유틸리티 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 웹캠 입력 설정
cap = cv2.VideoCapture(0)

# CSV 파일 경로 및 이름 설정
csv_file_path = 'hand_landmarks.csv'
file_exists = os.path.isfile(csv_file_path)

# CSV 파일 열기
with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # CSV 파일에 헤더 작성 (파일이 존재하지 않을 때만)
    if not file_exists or os.path.getsize(csv_file_path) == 0:
        headers = ['Timestamp', 'UserInput'] + \
                  [f'Left_Landmark_{i}_X' for i in range(21)] + [f'Left_Landmark_{i}_Y' for i in range(21)] + [f'Left_Landmark_{i}_Z' for i in range(21)]
        csv_writer.writerow(headers)

    # Hands 객체 생성
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        user_input = None
        capturing = False
        running = True

        while running:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # 이미지 전처리
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # 결과 후처리
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 손목을 기준으로 좌표를 계산하기 위한 데이터 초기화
            left_hand_x = [0.0] * 21
            left_hand_y = [0.0] * 21
            left_hand_z = [0.0] * 21

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[idx].classification[0].label
                    # 손목 점 (랜드마크 0)의 좌표 추출
                    wrist = hand_landmarks.landmark[0]
                    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
                    
                    # 각 랜드마크의 상대 좌표 계산
                    hand_x = [(landmark.x - wrist_x) for landmark in hand_landmarks.landmark]
                    hand_y = [(landmark.y - wrist_y) for landmark in hand_landmarks.landmark]
                    hand_z = [(landmark.z - wrist_z) for landmark in hand_landmarks.landmark]

                    # 카메라 좌우 반전을 고려하여 손의 좌우를 반대로 처리
                    if hand_label == 'Left' and len(hand_x) == 21:
                        # 왼손으로 인식된 경우 오른손으로 저장
                        left_hand_x = hand_x
                        left_hand_y = hand_y
                        left_hand_z = hand_z
                    elif hand_label == 'Right' and len(hand_x) == 21:
                        # 오른손으로 인식된 경우 왼손으로 저장
                        left_hand_x = hand_x
                        left_hand_y = hand_y
                        left_hand_z = hand_z

                    # 손목 점 (랜드마크 0)의 좌표를 화면에 그리기
                    wrist_x, wrist_y = int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0])

                    # 상대적 좌표 계산 및 축 그리기 (단위 벡터로 가정)
                    scale = 100  # 축의 크기
                    x_axis_end = (int(wrist_x + scale), wrist_y)
                    y_axis_end = (wrist_x, int(wrist_y - scale))
                    z_axis_end = (wrist_x, int(wrist_y - scale / 2))  # Z 축은 화면에서 깊이를 상징

                    # X 축 (빨간색)
                    cv2.line(image, (wrist_x, wrist_y), x_axis_end, (0, 0, 255), 2)
                    # Y 축 (초록색)
                    cv2.line(image, (wrist_x, wrist_y), y_axis_end, (0, 255, 0), 2)
                    # Z 축 (파란색)
                    cv2.line(image, (wrist_x, wrist_y), z_axis_end, (255, 0, 0), 2)

                    # 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            if capturing:
                current_time = datetime.now().strftime('%y%m%d%H%M%S')  # 현재 시간
                row = [current_time, user_input] + \
                      left_hand_x + left_hand_y + left_hand_z
                csv_writer.writerow(row)

            # 이미지 표시
            cv2.imshow('MediaPipe Hands with Coordinate Axes', cv2.flip(image, 1))

            # 키 입력 대기 (s: 시작, f: 종료, q: 프로그램 종료)
            key = cv2.waitKey(5) & 0xFF
            if key == ord('s'):  # 's' 키를 눌러 기록 시작
                user_input = int(input("기록할 정수를 입력하세요: "))
                capturing = True
                print("기록 시작")
            elif key == ord('f'):  # 'f' 키를 눌러 기록 종료 및 초기화
                capturing = False
                user_input = None
                print("기록 중지 및 사용자 입력 초기화")
            elif key == ord('q') or key == 27:  # 'q' 또는 'Esc' 키를 눌러 프로그램 종료
                running = False

# 자원 해제
cap.release()
cv2.destroyAllWindows()