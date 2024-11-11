from ultralytics import YOLO
import cv2
import numpy as np
import time


# Load model
model = YOLO('yolov8n.pt')
# result_img = model.predict("input_videos/image.png", save=True)
# print("Result Images: ", result_img)
# print("boxes")
# for box in result_img[0].boxes:
#     print(box)


result_video = model.track("input_videos/input_video.mp4", save=True)
# print("Result video: ", result_img)
# print("boxes")
# for box in result_video[0].boxes:
#     print(box)