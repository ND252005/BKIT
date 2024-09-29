from client_lib import GetStatus, GetRaw, GetSeg, AVControl ,CloseSocket
import cv2
import numpy as np
import math

CHECKPOINT = 150


#Toa do (0-179, 0-319)

def showImage(gray_show, center_row):
    h, w = gray_show.shape
    gray_show = cv2.line(gray_show, (center_row, CHECKPOINT), (int(w/2), h-1), 90, 2)
    gray_show = cv2.line(gray_show, (int(w/2), CHECKPOINT), (int(w/2), h-1), 90, 2)
    gray_show = cv2.line(gray_show, (int(w/2), CHECKPOINT), (center_row, CHECKPOINT), 90, 2)
    cv2.imshow('test', gray_show)

def AngCal(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray*(255/np.max(gray))).astype(np.uint8)
    gray_show = gray.copy()

    h, w = gray.shape

    line_row = gray[CHECKPOINT, :]
    line = np.where(line_row == 255)[0]

    min_x = line[0]
    max_x = line[-1]
    center_row = int((max_x+min_x + 1)/2)

    showImage(gray, center_row)

    if (max_x - min_x) == 319: #Xu ly nga tu
        return 0

    x0, y0 = int(w/2), h
    x1, y1 = center_row, CHECKPOINT

    value = (x1-x0)/(y0-y1)
    angle = math.degrees(math.atan(value)) / 3
    
    print("Value: {} --- Angle: {}".format(value, angle))

    if angle > 25:
        angle = 25
    elif angle < -25:
        angle = -25

    return angle

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()

            print(state)

            cv2.imshow('raw_image', raw_image)
            cv2.imshow('segment_image', segment_image)

            angle = AngCal(segment_image)  
            # print(angle)
            AVControl(speed = 20, angle = angle) # maxspeed = 90, max steering angle = 25

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        CloseSocket()