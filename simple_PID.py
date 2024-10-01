from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import math
import time


CHECKPOINT = 150
SPD_MIN = 15
SPD_MED = 25
SPD_MAX = 35
MAX_ANGLE = 25


# Tham số PID
Kp = 0.6  # Hệ số tỉ lệ (Proportional)
Ki = 0.01  # Hệ số tích phân (Integral)
Kd = 0.1  # Hệ số vi phân (Derivative)


# Kp = 0.6  # Giữ ở mức cân bằng
# Ki = 0.02  # Tăng nhẹ để giảm lỗi tích lũy
# Kd = 0.15  # Giữ ổn định




previous_error = 0
integral = 0


def showImage(gray_show, center_row):
   h, w = gray_show.shape
   cv2.line(gray_show, (center_row, CHECKPOINT), (int(w / 2), h - 1), 90, 2)
   cv2.line(gray_show, (int(w / 2), CHECKPOINT), (int(w / 2), h - 1), 90, 2)
   cv2.line(gray_show, (int(w / 2), CHECKPOINT), (center_row, CHECKPOINT), 90, 2)
   cv2.imshow('test', gray_show)


def AngCal(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


   h, w = gray.shape
   line_row = gray[CHECKPOINT, :]
   line = np.where(line_row == 255)[0]


   if len(line) == 0:
       return 0  # Không phát hiện đường


   min_x = line[0]
   max_x = line[-1]
   center_row = (max_x + min_x) // 2


   # Hiển thị ảnh
   showImage(gray, center_row)


   if (max_x - min_x) == 319:  # Xử lý ngã tư
       return 0


   # Tính toán góc lái
   x0, y0 = w // 2, h
   x1, y1 = center_row, CHECKPOINT
   value = (x1 - x0) / (y0 - y1)
   angle = math.degrees(math.atan(value)) / 3


   # Hạn chế góc lái trong khoảng -25 đến 25 độ
   angle = max(min(angle, MAX_ANGLE), -MAX_ANGLE)


   return angle


def pid_controller(angle, dt):
   global previous_error, integral


   # Tính toán PID
   error = angle
   integral += error * dt
   derivative = (error - previous_error) / dt


   output = Kp * error + Ki * integral + Kd * derivative
   previous_error = error


   # Hạn chế đầu ra của góc lái
   output = max(min(output, MAX_ANGLE), -MAX_ANGLE)


   return output


if __name__ == "__main__":
   try:
       prev_time = time.time()
       while True:
           state = GetStatus()
           segment_image = GetSeg()


           # Tính toán góc lái
           angle = AngCal(segment_image)


           # Tính thời gian delta để PID hoạt động
           current_time = time.time()
           dt = current_time - prev_time
           prev_time = current_time


           # Sử dụng PID để tính góc lái
           pid_angle = pid_controller(angle, dt)


           # Điều chỉnh tốc độ theo góc lái PID
           if abs(pid_angle) < 5:
               speed = SPD_MAX  # Tăng tốc khi đánh lái nhẹ
           elif abs(pid_angle) < 15:
               speed = SPD_MED  # Tốc độ trung bình khi góc lái vừa
           else:
               speed = SPD_MIN  # Giảm tốc khi đánh lái mạnh


           print(f"Speed: {speed}, PID Angle: {pid_angle:.2f}")
           AVControl(speed=speed, angle=pid_angle)  # Tốc độ tối đa là 90, góc lái tối đa là 25


           key = cv2.waitKey(1)
           if key == ord('q'):
               break
   finally:
       CloseSocket()
