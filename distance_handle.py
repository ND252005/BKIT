from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import math
import time


CHECKPOINT = 150
speed_thresholds = [
    (78, 55),  # (LEVEL_1, SPD_LEVEL_1)
    (73, 40),  # (LEVEL_2, SPD_LEVEL_2)
    (65, 35),  # (LEVEL_3, SPD_LEVEL_3)
    (60, 30),  # (LEVEL_4, SPD_LEVEL_4)
    (55, 25),  # (LEVEL_5, SPD_LEVEL_5)
    (40, 20),  # (LEVEL_6, SPD_LEVEL_6)
    (30, 15),  # (LEVEL_7, SPD_LEVEL_7)
    (25, 10),  # (LEVEL_8, SPD_LEVEL_8)
]
MAX_ANGLE = 25
#LANE_THRESHOLD = 300  # Ngưỡng để ưu tiên bên phải
acceleration = 0


# Tham số PID
Kp = 0.8
Ki = 0.017
Kd = 0.3


previous_error = 0
integral = 0


def showImage(gray_show, center_row):
   h, w = gray_show.shape
   cv2.line(gray_show, (center_row, CHECKPOINT), (int(w / 2), h - 1), 90, 2)
   cv2.line(gray_show, (int(w / 2), CHECKPOINT), (int(w / 2), h - 1), 90, 2)
   cv2.line(gray_show, (int(w / 2), CHECKPOINT), (center_row, CHECKPOINT), 90, 2)
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

def calculate_speed(distance_to_black):
    # Duyệt qua danh sách từ trên xuống, nếu vượt qua ngưỡng thì lấy tốc độ tương ứng
    for level, speed in speed_thresholds:
        if distance_to_black >= level:
            return speed
    return 7


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
   value = (x1 - x0) / (y0 - y1)
   angle = math.degrees(math.atan(value)) / 3


   angle = max(min(angle, MAX_ANGLE), -MAX_ANGLE)


   return angle, center_row, gray


def pid_controller(angle, dt):
   global previous_error, integral


   error = angle
   integral += error * dt
   derivative = (error - previous_error) / dt

   output = Kp * error + Ki * integral + Kd * derivative
   previous_error = error


   output = max(min(output, MAX_ANGLE), -MAX_ANGLE)


   return output


if __name__ == "__main__":
   try:
       speed_check_time = time.time()  # Thời gian lần cuối kiểm tra vận tốc
       prev_speed = None  # Biến lưu trữ tốc độ trước đó
       prev_time = time.time()
       while True:
           state = GetStatus()
           segment_image = GetSeg()


           # Lấy tốc độ thực tế của xe từ state
           current_speed = state['Speed']
           print(f"Actual Speed: {current_speed}")


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
           speed = calculate_speed(distance_to_black)


           print(f"Speed: {speed}, PID Angle: {pid_angle:.2f}")
           AVControl(speed=speed, angle=pid_angle)


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



