import cv2
import numpy as np
import torch

from src.keypoint import Keypoint

if __name__ == "__main__":
    model_path = "model/pre_model_3000.pth"
    keypoint = Keypoint(model_path, threshold=0.2)

    img = cv2.imread("test_data/standing.jpg")

    all_peaks = keypoint(img)

    # 带有score的关键点
    for i, peaks in enumerate(all_peaks):
        for peak in peaks:
            x, y, score = peak
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)
            # 编号
            cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # 可信度
            # cv2.putText(img, f"{score:.2f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("img", img)
    cv2.waitKey(0)