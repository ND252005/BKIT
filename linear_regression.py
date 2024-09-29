from client_lib import GetStatus, GetRaw, GetSeg, AVControl ,CloseSocket
import cv2
import numpy as np
import math
import time

CHECKPOINT = 120
SPD_LEVEL_1 = 55  # Tốc độ cao nhất
SPD_LEVEL_2 = 45  # Tốc độ mức 2
SPD_LEVEL_3 = 42  # Tốc độ mức 3
SPD_LEVEL_4 = 35  # Tốc độ chậm nhất
SPD_LEVEL_5 = 32
SPD_LEVEL_6 = 24 
SPD_LEVEL_7 = 18 
SPD_LEVEL_8 = 15 
SPD_LEVEL_9 = 10 
LEVEL_1 = 48
LEVEL_2 = 43
LEVEL_3 = 35
LEVEL_4 = 30
LEVEL_5 = 25
LEVEL_6 = 18
LEVEL_7 = 10
LEVEL_8 = 5

MAX_ANGLE = 25
#LANE_THRESHOLD = 300  # Ngưỡng để ưu tiên bên phải
acceleration = 0

# Tham số PID
Kp = 0.8
Ki = 0.001
Kd = 0.002

previous_error = 0
integral = 0

def showImage(gray_show, center_row):
    h, w = gray_show.shape

    dy = 50
    gray_show = cv2.copyMakeBorder(gray_show, 0, dy, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.line(gray_show, (center_row, CHECKPOINT), (int(w / 2), h - 1 + dy), 90, 2)
    cv2.line(gray_show, (int(w / 2), CHECKPOINT), (int(w / 2), h - 1 + dy), 90, 2)
    cv2.line(gray_show, (int(w / 2), CHECKPOINT), (center_row, CHECKPOINT), 90, 2)

    xx = 50

    cv2.imshow('test', gray_show)

def calculate_distance_to_black(image, center_row, checkpoint):
    """
    Tính khoảng cách từ điểm (center_row, checkpoint) đến pixel đen gần nhất trên cột center_row.
    """
    h, _ = image.shape
    
    # Lấy tất cả các giá trị pixel trên cột center_row
    column = image[:checkpoint, center_row]  # Lấy từ trên đến checkpoint
    
    # Tìm các pixel màu đen (giá trị = 0) trong cột
    black_pixels = np.where(column == 0)[0]  # Trả về index của các pixel đen
    
    if len(black_pixels) == 0:
        return h  # Nếu không tìm thấy pixel đen nào, trả về giá trị lớn nhất (coi như không có chướng ngại)
    
    # Tính khoảng cách từ checkpoint đến pixel đen gần nhất
    closest_black_pixel = black_pixels[-1]  # Lấy pixel đen cao nhất (gần checkpoint nhất)
    distance = checkpoint - closest_black_pixel  # Tính khoảng cách

    return distance

def AngCal(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    h, w = gray.shape
    line_row = gray[CHECKPOINT, :]
    line = np.where(line_row == 255)[0]

    if len(line) == 0:
        return None  # Không cần xử lý lùi, bỏ qua khi không thấy đường

    min_x = line[0]
    max_x = line[-1]
    lane_width = max_x - min_x

    # Ưu tiên đi về làn bên phải khi có thể
    
    # if lane_width < LANE_THRESHOLD:
    #     center_row = int((2.5 * max_x + min_x) / 3.5)
    # else:
    #     center_row = (max_x + min_x) // 2

    center_row = (max_x + min_x) // 2

    showImage(gray, center_row)

    # Tính toán góc lái
    x0, y0 = w // 2, h
    x1, y1 = center_row, CHECKPOINT
    value = (x1 - x0) / (abs(y0 - y1 )+ 70)
    angle = math.degrees(math.atan(value)) / 3

    angle = max(min(angle, MAX_ANGLE), -MAX_ANGLE)

    return angle, center_row, gray

def pid_controller(angle, dt):
    global previous_error, integral

    error = angle

    integral += error * dt
    max_I_error = 20
    integral = max(integral, -max_I_error)
    integral = min(integral, max_I_error)

    derivative = (error - previous_error) / dt
    

    output = Kp * error + Ki * integral + Kd * derivative
    previous_error = error

    output = max(min(output, MAX_ANGLE), -MAX_ANGLE)

    return output


def linear_regression(x, y):
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Number of points
    n = len(x)
    
    # Calculate slope (a) and intercept (b)
    a = (n * np.sum(x*y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
    b = (np.sum(y) - a * np.sum(x)) / n
    

    return a, b

def process_central(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    


    height, width = image.shape
    dy = 3
    
    image[:height//2] = 0
    img = image[height//2:height*4//5]
    total = 0
    count = 0
    for yyyy in img:
        for xxx in range(width):
            if yyyy[xxx] > 100:
                count += xxx
                total += 1
    if total == 0:
        total = 1

    values = list()
    y_list = list()
    for i in range(30):
        try:
            yy = height-15-i*dy
            line = np.where(image[yy, :] == 255)[0]
            value = (line[0]+line[-1])//2

            
            if values and abs(value - values[-1]) > 20:
                continue

            image[yy, value] = 0
            y_list.append(yy)
            values.append(round(value))
        except:
            pass
        
    # print(values)
    cv2.imshow("anhpn", image)
    threshold = 10
    if len(y_list) > threshold:
        y_list = y_list[:threshold]
        values = values[:threshold]
    a, b = linear_regression(y_list, values)
    if a > 50:
        a = 50
    if a < -50:
        a = -50

    angle = 0
    # angle = round(np.degrees(np.arctan(-a)))
    # angle = max(-MAX_ANGLE*2, min(MAX_ANGLE*2, angle))

    # zzz = 0.6
    # a_ = max(min(-1/a, zzz), -zzz)
    # angle = round(np.degrees(np.arctan(a_))
    error = count/total-160
    print(a, b, angle, error)
    # return angle
    return error

if __name__ == "__main__":
    try:
        speed_check_time = time.time()  # Thời gian lần cuối kiểm tra vận tốc
        prev_speed = None  # Biến lưu trữ tốc độ trước đó
        prev_time = time.time()
        while True:
            raw_image = GetRaw()
            cv2.imshow('raw_image', raw_image)
            # cv2.imshow('raw_blue', raw_image[:, :, 0])
            # cv2.imshow('raw_green', raw_image[:, :, 1])
            # cv2.imshow('raw_red', raw_image[:, :, 2])

            state = GetStatus()
            segment_image = GetSeg()
            anh_angle = process_central(segment_image)

            # Lấy tốc độ thực tế của xe từ state
            current_speed = state['Speed']
            # current_speed = state['Angle']
            print(f"Actual Speed: {state}")

            result = AngCal(segment_image)

            if result is None:
                continue

            angle, center_row, gray_image = result

            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            pid_angle = pid_controller(angle, dt)

            # Tính khoảng cách từ (center_row, CHECKPOINT) đến pixel màu đen gần nhất
            distance_to_black = calculate_distance_to_black(gray_image, center_row, CHECKPOINT)
            print(f"Distance to Black: {distance_to_black}")

            # Điều chỉnh tốc độ theo khoảng cách đến pixel đen gần nhất
            if distance_to_black >= LEVEL_1:
                speed = SPD_LEVEL_1  # Chạy với tốc độ cao nhất
            elif LEVEL_2 <= distance_to_black < LEVEL_1:
                speed = SPD_LEVEL_2  # Chạy với tốc độ mức 2
            elif LEVEL_3 <= distance_to_black < LEVEL_2:
                speed = SPD_LEVEL_3 
            elif LEVEL_4 <= distance_to_black < LEVEL_3:
                speed = SPD_LEVEL_4
            elif LEVEL_5 <= distance_to_black < LEVEL_4:
                speed = SPD_LEVEL_5
            elif LEVEL_6 <= distance_to_black < LEVEL_5:
                speed = SPD_LEVEL_6
            elif LEVEL_7 <= distance_to_black < LEVEL_6:
                speed = SPD_LEVEL_7
            elif LEVEL_8 <= distance_to_black < LEVEL_7:
                speed = SPD_LEVEL_8
            else:
                speed = SPD_LEVEL_9  # Chạy chậm nhất khi gặp đoạn hẹp hoặc cua gấp

            print(f"Speed: {speed}, PID Angle: {pid_angle:.2f}")
            AVControl(speed=speed*0.9,  angle=pid_angle)

            if current_time - speed_check_time >= 0.5:  # Sau mỗi 2 giây
                if prev_speed is not None:
                    acceleration = (current_speed - prev_speed) * 2  # Tính gia tốc
                prev_speed = current_speed
                speed_check_time = current_time  # Cập nhật thời gian kiểm tra
            
            print(f"Acceleration: {acceleration} m/s²")

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        CloseSocket()
