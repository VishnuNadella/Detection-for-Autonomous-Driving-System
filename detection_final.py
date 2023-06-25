import cv2
from ultralytics import YOLO
import numpy as np
import time
from PIL import ImageGrab


model = YOLO("yolov8s-seg.pt")


def roi(img,vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask,vertices,255) 
    masked_image=cv2.bitwise_and(img,mask)
    return masked_image

def preprocess(img):
    roi_vertices = ([0, 390], [0, 370], [400, 280], [700, 280], [1060, 370], [1060, 390])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    canny = cv2.Canny(gray, 250, 300)

    cropped = roi(canny, np.array([roi_vertices], np.int32))
    cv2.imshow("ROI", cropped)

    _, thresholded = cv2.threshold(cropped, 127, 255, cv2.THRESH_BINARY)

    return thresholded


def draw_hough_lines(img,lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(blank_image,(x1,y1),(x2,y2),(255,0,255),thickness=10)

        img = cv2.addWeighted(img,0.8,blank_image,1,0.0)
    else:
        print("No Lanes detected")
    return img


for i in list(range(3))[::-1]:
    print(i+1)
    time.sleep(1)

while True:
    frame = np.array(ImageGrab.grab(bbox=(0, 40, 1060, 650))) # For FH4
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cropped= preprocess(frame)
    lines = cv2.HoughLinesP(cropped, rho=6, threshold=60, theta=np.pi/180, minLineLength=50, maxLineGap=150, lines=np.array([]))

    img = draw_hough_lines(frame,lines)
    results = model(source = img, stream = True, show = True, conf = 0.45)

    cv2.imshow('Lane detection',img)

    for result in results:
        boxes = result.boxes
        masks = result.masks
        class_ids = result.names

    if cv2.waitKey(1) & 0xFF == 27:
        break