import time

from pynput.keyboard import Key, Controller, Listener
import cv2
import dxcam

import win32api as wapi
import win32con

from src.keypoint import Keypoint
from src.utils import util

cx = wapi.GetSystemMetrics(0) // 2
cy = wapi.GetSystemMetrics(1) // 2

print("Screen size: ", cx*2, cy*2)

# 屏幕中心256*256的区域
detect_region = (cx - 128, cy - 128, cx + 128, cy + 128)

detect_on = True
end = False

def update_v(x, y):
    # print('move rel:', x, y)
    wapi.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x), int(y))

model_path = "model/pre_model_3000.pth"
keypoint_estimation = Keypoint(model_path, half=False, threshold=0.2)

#异步监听CAPSLOCK键，切换detect_on的值
def on_press(key):
    global detect_on
    if key == Key.caps_lock:
        detect_on = not detect_on
        print('detect_on:', detect_on)

listener = Listener(on_press=on_press)
listener.start()

camera = dxcam.create(region=detect_region, output_color='BGR')

frame = 0
fps = 0.0
last_move_frame = 0

# test_x = 0
# test_dx = [None] * 256

t = time.perf_counter()

while True:
    start_detect = time.perf_counter()
    new_img = camera.grab()
    if new_img is not None:
        img = new_img
    # print('grab eclipsed:', time.perf_counter() - start_detect)

    if not detect_on:
        # 显示检测已关闭
        cv2.putText(img, 'detect off', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Detect Screen', img)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        continue
    
    all_peaks = keypoint_estimation(img)
    # print('keypoint eclipsed:', time.perf_counter() - start_detect)
    canvas = img.copy()
    canvas = util.draw_keypoint(canvas, all_peaks)
    curx, cury = wapi.GetCursorPos()[0], wapi.GetCursorPos()[1]
    # 找到离鼠标最近的关键点，坐标是相对于detect_region的
    nose = util.find_near_keypoint(all_peaks, curx - detect_region[0], cury - detect_region[1])
    if nose is not None:

        x, y = nose[0], nose[1]
        # 目标的位置与鼠标的偏差
        dx, dy = (x + detect_region[0]) - curx, (y + detect_region[1]) - cury

        # 测试鼠标移动的距离
        # if test_x < 256:
        #     if test_dx[test_x] is None:
        #         test_dx[test_x] = dx
        #         print('test_dx:', test_x, dx)
        #     elif abs(dx) < 1.0:
        #         test_x += 1
        #         update_v(test_x, 0)
        #         time.sleep(0.05)
        #         continue

        dx, dy = dx * 2, dy * 2
        # 移动鼠标，要等上一次移动完成
        if (dx != 0 or dy != 0) and frame - last_move_frame > 1:
            update_v(dx, dy)
            last_move_frame = frame

    # 为了性能，每10帧显示一次
    if frame % 10 == 0:
        # 标题显示fps（小数点后2位）
        cv2.putText(canvas, 'fps: {:.2f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Detect Screen', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    fps = 0.9 * fps + 0.1 * (1. / (time.perf_counter() - t))
    t = time.perf_counter()
    # if frame % 20 == 0: print(fps)
    frame += 1

cv2.destroyAllWindows()
listener.stop()
end = True

# print(test_dx)