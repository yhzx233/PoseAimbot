import cv2

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def draw_keypoint(canvas, all_peaks):

    for i in range(len(all_peaks)):
        for x, y, _ in all_peaks[i]:
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

# 找最近的关键点，先按照编号，再按照距离，返回关键点坐标
def find_near_keypoint(all_peaks, x, y):
    min_dist = 1e10
    min_keypoint = None
    for i in range(len(all_peaks)):
        for keypoint in all_peaks[i]:
            dist = (keypoint[0] - x) ** 2 + (keypoint[1] - y) ** 2
            if dist < min_dist:
                min_dist = dist
                min_keypoint = keypoint
        if min_keypoint is not None:
            return min_keypoint
    return None