from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note
import cv2
import os
from datetime import datetime
import time
import shutil

from model import get_angel

robot = Create3(Bluetooth())


def capture_and_save_image():
    base_path = '/Users/narekharutyunyan/Desktop/RoboChaser/Model'
    folder_path = os.path.join(base_path, 'data')

    cap = cv2.VideoCapture(1)  

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    ret, frame = cap.read()

    cap.release()

    if not ret:
        print("Error: Failed to capture image.")
        return

    if frame is None or not frame.any():
        print("Error: Captured image is empty or black.")
        return

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:    
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path) 
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path) 
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    filename = datetime.now().strftime("%Y%m%d%H%M%S") + ".png"
    file_path = os.path.join(folder_path, filename)

    # frame = frame[152:568, 432:848]

    cv2.imwrite(file_path, frame)


def get_angle_change():
    capture_and_save_image()
    return get_angel()


@event(robot.when_play)
async def play(robot):
    while True:
        turn_angle, is_object = get_angle_change()
        if is_object:
            await robot.turn_left(turn_angle)
            await robot.move(15)
        else:
            await robot.turn_left(turn_angle)  

robot.play()









