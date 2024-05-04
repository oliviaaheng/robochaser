from ultralytics import YOLO
import os
import random

model = YOLO('best.pt')

data_path = '/Users/narekharutyunyan/Desktop/RoboChaser/Model/data'

def get_angel():
    global data_path

    files = os.listdir(data_path)
    image_file = next(file for file in files if file.endswith('.png'))
    image_path = os.path.join(data_path, image_file)


    results = model.predict(image_path, save=True)

    try:
        x = float(results[0].boxes.xywhn[0][0])
        return (calculate_turn_angle(x=x), True)
    except:
        print("No object")
        return (random.randint(-30, 30), False)


def calculate_turn_angle(x, max_angle=30):
    difference = 0.5 - x
    angle = difference * 2 * max_angle

    return angle

    
# print(calculate_turn_angle(x=0))
    