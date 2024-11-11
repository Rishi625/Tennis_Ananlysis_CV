from ultralytics import YOLO
import cv2
import numpy as np
import time


# Load model
model = YOLO('Models/best.pt')
result_yoloc5s = model.predict("input_videos/image.png",conf = 0.2, save=True)
print("Result Images: ", result_yoloc5s)
print("boxes")
for box in result_yoloc5s[0].boxes:
    print(box)

result_yoloc5s_video = model.predict("input_videos/input_video.mp4",conf = 0.2, save=True)
print("Result video: ", result_yoloc5s_video)
print("boxes")
for box in result_yoloc5s_video[0].boxes:
    print(box)