from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import math
import time

CHECKPOINT_1 = 160
CHECKPOINT_2 = 145  # Checkpoint thứ hai
MAX_ANGLE = 25
LANE_THRESHOLD = 250
acceleration = 0

# Mảng chứa các mức tốc độ tương ứng với góc đánh lái
angle_speed_thresholds = [
    (3, 55),   # (Góc nhỏ, tốc độ cao)
    (7, 40),  # (Góc trung bình, tốc độ trung bình)
    (16, 35),  # (Góc lớn hơn, tốc độ giảm)
    (20, 15),  # (Góc lớn, tốc độ giảm nhiều)
    (25, 7)   # (Góc rất lớn, tốc độ thấp nhất)
]

# Mảng chứa các mức tốc độ tương ứng với khoảng cách
distance_speed_thresholds = [
    (90, 55),   # (Khoảng cách lớn, tốc độ cao)
    (40, 45),
    (25, 25),
    (15, 10),
    (5, 7)   # (Khoảng cách nhỏ, tốc độ thấp)
]

# Tham số PID
Kp = 0.8
Ki = 0.01
Kd = 0.2

previous_error = 0
integral = 0

def showImage(gray_show, center_row_1, center_row_2):
    h, w = gray_show.shape
    # Hiển thị checkpoint 1
    cv2.line(gray_show, (center_row_1, CHECKPOINT_1), (int(w / 2), h - 1), 90, 2)
    cv2.line(gray_show, (int(w / 2), CHECKPOINT_1), (int(w / 2), h - 1), 90, 2)
    cv2.line(gray_show, (int(w / 2), CHECKPOINT_1), (center_row_1, CHECKPOINT_1), 90, 2)
    
    # Hiển thị checkpoint 2
    cv2.line(gray_show, (center_row_2, CHECKPOINT_2), (int(w / 2), h - 1), 90, 2)
    cv2.line(gray_show, (int(w / 2), CHECKPOINT_2), (int(w / 2), h - 1), 90, 2)
    cv2.line(gray_show, (int(w / 2), CHECKPOINT_2), (center_row_2, CHECKPOINT_2), 90, 2)

    cv2.imshow('test', gray_show)

def calculate_angle_and_lane_width(image, checkpoint):
    """
    Tính toán góc lái và chiều rộng của làn đường tại một checkpoint cụ thể.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    h, w = gray.shape
    line_row = gray[checkpoint, :]
    line = np.where(line_row == 255)[0]

    if len(line) == 0:
        return None, None, None  # Không thấy đường

    min_x = line[0]
    max_x = line[-1]
    lane_width = max_x - min_x

    if lane_width < LANE_THRESHOLD:
        center_row = int((2.2 * max_x + min_x) / 3.2)
    else:
        center_row = (max_x + min_x) // 2

    # Tính toán góc lái
    x0, y0 = w // 2, h
    x1, y1 = center_row, checkpoint
    value = (x1 - x0) / (y0 - y1)
    angle = math.degrees(math.atan(value)) / 3

    angle = max(min(angle, MAX_ANGLE), -MAX_ANGLE)

    return angle, center_row, lane_width

def calculate_distance_to_black(image, center_row, checkpoint):
   """
   Tính khoảng cách từ điểm (center_row, checkpoint) đến pixel đen gần nhất trên cột center_row.
   """
   h, _, _ = image.shape
  
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


def pid_controller(angle, dt):
    global previous_error, integral

    error = angle
    integral += error * dt
    derivative = (error - previous_error) / dt

    output = Kp * error + Ki * integral + Kd * derivative
    previous_error = error

    output = max(min(output, MAX_ANGLE), -MAX_ANGLE)

    return output

def calculate_speed_from_angle(angle):
    # Lấy tốc độ tương ứng dựa trên góc đánh lái
    for angle_threshold, speed in angle_speed_thresholds:
        if abs(angle) <= angle_threshold:
            return speed
    return 5  # Tốc độ thấp nhất mặc định nếu góc lớn hơn mức cao nhất

def calculate_speed_from_distance(distance_to_black):
    # Duyệt qua danh sách từ trên xuống, nếu vượt qua ngưỡng thì lấy tốc độ tương ứng
    for level, speed in distance_speed_thresholds:
        if distance_to_black >= level:
            return speed
    return 7

# Code điều khiển chính
if __name__ == "__main__":
    try:
        prev_time = time.time()

        while True:
            state = GetStatus()
            segment_image = GetSeg()

            # Lấy tốc độ thực tế của xe từ state
            current_speed = state['Speed']
            print(f"Actual Speed: {current_speed}")

            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            # Tính toán góc lái và chiều rộng làn đường tại hai checkpoint
            angle_1, center_row_1, lane_width_1 = calculate_angle_and_lane_width(segment_image, CHECKPOINT_1)
            angle_2, center_row_2, lane_width_2 = calculate_angle_and_lane_width(segment_image, CHECKPOINT_2)

            if angle_1 is None and angle_2 is None:
                continue  # Bỏ qua nếu không phát hiện được làn đường tại cả hai checkpoint

            # Tính toán khoảng cách từ checkpoint 2 đến điểm đen
            distance_to_black = calculate_distance_to_black(segment_image, center_row_2, CHECKPOINT_2)
            print(f"Distance to black spot: {distance_to_black}")

            # Ưu tiên checkpoint có lane_width nhỏ hơn
            if lane_width_1 is not None and lane_width_2 is not None:
                if lane_width_1 <= lane_width_2:
                    chosen_angle = angle_1
                    chosen_center_row = center_row_1
                    speed_from_angle = calculate_speed_from_angle(angle_1)
                else:
                    chosen_angle = angle_2
                    chosen_center_row = center_row_2
                    speed_from_angle = calculate_speed_from_angle(angle_2)
            elif lane_width_1 is not None:
                chosen_angle = angle_1
                chosen_center_row = center_row_1
                speed_from_angle = calculate_speed_from_angle(angle_1)
            else:
                chosen_angle = angle_2
                chosen_center_row = center_row_2
                speed_from_angle = calculate_speed_from_angle(angle_2)

            # Hiển thị cả hai checkpoint lên ảnh
            showImage(cv2.normalize(cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY), None, 0, 255, cv2.NORM_MINMAX), center_row_1, center_row_2)

            # Tính tốc độ dựa trên khoảng cách đến pixel đen tại checkpoint 2
            speed_from_distance = calculate_speed_from_distance(distance_to_black)

            # Chọn tốc độ thấp hơn giữa speed_from_angle và speed_from_distance
            final_speed = min(speed_from_angle, speed_from_distance)

            # Điều chỉnh PID dựa trên góc đánh lái
            pid_angle = pid_controller(chosen_angle, dt)
            print(speed_from_angle, speed_from_distance)

            print(f"Final Speed: {final_speed}, PID Angle: {pid_angle:.2f}")
            AVControl(speed=final_speed, angle=pid_angle)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    finally:
        CloseSocket()
