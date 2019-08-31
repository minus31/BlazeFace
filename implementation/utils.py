# Anchors : scale factor == that of SSD  : 8 of scale levels
# - 16x16 : [0.2, sqrt(0.2 * 0.43)]
# -  8x8  : [0.43, sqrt(0.43 * 0.67), sqrt(0.67, 0.9), 0.9, sqrt(0.9 * 1)]

# get IoU 
import numpy as np

def get_iou(p1, p2):
    """
    get Jaccard overlap(IoU) value 

    p1 = [x, y, w, h] :: x, y in [0, 1]
    p2 = [x, y, w, h] :: x, y in [0, 1]

    return : IoU 
    """

    """converting to [left-bottom coord, right-top coord]"""

    # in my case image-size = 128,128

    p1[0] *= 128.
    p1[1] *= 128.
    p2[0] *= 128.
    p2[1] *= 128.

    box1 = [p1[0] - (p1[2] // 2), p1[1] - (p1[3] // 2),
            p1[0] + (p1[2] // 2), p1[1] + (p1[3] // 2)]
    box2 = [p2[0] - (p2[2] // 2), p2[1] - (p2[3] // 2),
            p2[0] + (p2[2] // 2), p2[1] + (p2[3] // 2)]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = p1[2] * p1[3]
    boxBArea = p2[2] * p2[3]

    if interArea == 0:
        return 0.

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou    


if __name__ == "__main__":
    pass

