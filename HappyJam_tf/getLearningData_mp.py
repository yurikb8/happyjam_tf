import cv2
import mediapipe as mp
import csv
from collections import deque
import time
import argparse
import itertools

#コマンドライン引数
parser = argparse.ArgumentParser()
parser.add_argument("--gesture_id", type=int, default=0) #ラベル種別 0=静止, 1=演奏
parser.add_argument("--time", type=int, default=10) #録画時間(sec)
args = parser.parse_args()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#csvのパス
csv_path = './point_history.csv'

start_time = time.time()

tmp_landmarks_list=[]

# 座標履歴を保持するための変数
history_length = 8
point_history = deque(maxlen=history_length)

# CSVファイルに座標履歴を保存する関数
def logging_csv(gesture_id, csv_path, point_history_list):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gesture_id, *point_history_list])
    return

# For webcam input:
filepath = "./movie/test2.mp4"
cap = cv2.VideoCapture(filepath) #引数0ならカメラ,filepathで動画
with mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    if results.pose_landmarks:
        tmp_landmarks_list.append([results.pose_landmarks.landmark[0].x,results.pose_landmarks.landmark[0].y])

        for idx in range(11,23):
            tmp_landmarks_list.append([results.pose_landmarks.landmark[idx].x,results.pose_landmarks.landmark[idx].y])

        # 人差指の指先座標を履歴に追加
        point_history.append(tmp_landmarks_list)

    if len(point_history) == history_length:
        point_history_list = list(itertools.chain.from_iterable(point_history))
        logging_csv(args.gesture_id, csv_path,point_history_list)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == 27:
        break

    if (time.time() - start_time) > args.time:
        break

cap.release()
