"""
Adapted from https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

animate = True

# Ground truth boxes (NOT USED)
boxes = np.array([[323.4543, 93.6051, 638.6754, 423.2758],
                  [10.6155, 193.5905, 235.0694, 368.0622],
                  [312.1205, 103.2962, 381.5783, 220.8106],
                  [210.9172, 105.3266, 333.4616, 218.0198]])
scores = np.array(([0.9981, 0.9946, 0.9867, 0.9198]))

# Proposals
proposals_boxes = np.array(([[323.4543, 93.6051, 638.6754, 423.2758],
                             [303.2942, 97.5246, 631.9091, 417.0857],
                             [3.8024, 189.8524, 225.6536, 382.5255],
                             [199.6172, 107.7098, 337.2172, 217.5047],
                             [311.1430, 99.5802, 380.6906, 226.1043],
                             [340.8973, 90.8514, 639.1058, 415.3783],
                             [276.6788, 100.8728, 631.2170, 426.0000],
                             [310.5667, 100.6068, 640.0000, 425.9723],
                             [309.3228, 98.5854, 383.7373, 228.9948],
                             [332.3773, 81.3674, 638.7166, 422.3216],
                             [4.8267, 192.0031, 230.1411, 371.0965],
                             [195.4081, 106.3627, 326.8040, 211.8499],
                             [308.9802, 98.0432, 384.2322, 227.4321],
                             [306.3264, 94.4237, 640.0000, 419.3774],
                             [311.9501, 98.5913, 384.7763, 225.2450],
                             [325.5839, 96.4924, 640.0000, 426.0000],
                             [317.5015, 95.8613, 640.0000, 416.1939],
                             [5.5200, 193.7063, 231.9213, 373.5116],
                             [313.5414, 96.4146, 640.0000, 420.6308],
                             [210.9172, 105.3266, 333.4616, 218.0198],
                             [310.3091, 99.1141, 377.9840, 226.6312],
                             [2.9152, 197.0478, 227.4817, 371.7160],
                             [325.3222, 96.7492, 640.0000, 416.3512],
                             [309.7229, 100.2535, 378.6619, 222.6704],
                             [0.0000, 184.1680, 230.0177, 382.5576],
                             [321.2011, 84.5259, 634.3634, 426.0000],
                             [309.8573, 99.4529, 380.1679, 226.1566],
                             [338.7589, 98.0144, 639.8776, 422.3615],
                             [311.7276, 100.3284, 383.1776, 227.4537],
                             [1.0550, 183.6293, 230.1038, 379.5150],
                             [8.7431, 195.0188, 231.1397, 365.4008],
                             [0.0000, 184.5661, 218.0917, 369.1828],
                             [312.1205, 103.2962, 381.5783, 220.8106],
                             [310.3988, 99.2471, 381.9437, 224.3915],
                             [10.6155, 193.5905, 235.0694, 368.0622],
                             [3.5149, 192.8371, 227.1362, 373.9926],
                             [0.0000, 195.3582, 227.0703, 390.9271],
                             [5.6514, 192.4634, 230.9048, 376.8218]]))

proposals_score = np.array(([0.9981, 0.9925, 0.9875, 0.9127, 0.9802, 0.9351, 0.9445, 0.9924, 0.9861,
                             0.9880, 0.9861, 0.9125, 0.9827, 0.9289, 0.9841, 0.9916, 0.9963, 0.9615,
                             0.9685, 0.9198, 0.9374, 0.9679, 0.9496, 0.9377, 0.9543, 0.9871, 0.9199,
                             0.9256, 0.9717, 0.9828, 0.9938, 0.9892, 0.9867, 0.9521, 0.9946, 0.9174,
                             0.9317, 0.9265]))


def draw_boxes(image, boxes, color=(0, 255, 0)):
    img = image.copy()
    for box in boxes:
        x0, y0, x1, y1 = box.astype(np.int64)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=2)
    return img


def animate_hook(image, proposals, selected_box, selected_score, keep, threshold, iteration):
    # Draw proposals and selected
    img = draw_boxes(image, proposals)
    img = draw_boxes(img, [selected_box], color=(255, 0, 0))
    img = draw_boxes(img, keep, color=(0, 0, 255))

    plt.title(
        f'Iteration = {iteration}\nRed: selected,  Score: {selected_score}\nBlue = Keep\nIoU Threshold = {threshold}')
    plt.imshow(img)
    plt.tight_layout()
    plt.pause(3)
    plt.cla()


def nms(proposals, score, threshold=0.5):
    """

    Args:
        proposals (N, 4):       Proposals
        score (N):              Proposals' score [0, 1]
        threshold:              IoU threshold

    Returns:
    """
    xmin, ymin, xmax, ymax = proposals.T

    keep_boxes = []
    keep_scores = []

    # Compute areas of bounding boxes
    areas = (xmax - xmin) * (ymax - ymin)

    # Sort by confidence score
    order = np.argsort(score)

    # Iterate bounding boxes
    iteration = 1
    while order.size > 0:

        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        selected_box, selected_score = proposals[index], score[index]

        # Compute intersection-over-union(IOU) against selected box
        x1 = np.maximum(xmin[index], xmin[order[:-1]])
        x2 = np.minimum(xmax[index], xmax[order[:-1]])
        y1 = np.maximum(ymin[index], ymin[order[:-1]])
        y2 = np.minimum(ymax[index], ymax[order[:-1]])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        iou = intersection / (areas[index] + areas[order[:-1]] - intersection)

        if animate:
            animate_hook(image, proposals[order], selected_box, selected_score, keep_boxes, threshold, iteration)

        # Suppress boxes with iou greater than threshold
        left = np.where(iou < threshold)
        order = order[left]

        keep_boxes.append(selected_box)
        keep_scores.append(selected_score)
        iteration += 1

    if animate:
        animate_hook(image, proposals[order], selected_box, selected_score, keep_boxes, threshold, iteration)
    else:
        out = draw_boxes(image, keep_boxes, color=(0, 0, 255))
        plt.imshow(out)
    plt.show()
    return keep_boxes, keep_scores


if __name__ == '__main__':
    # Image name
    image_name = 'image.jpg'

    # Bounding boxes
    bounding_boxes = proposals_boxes
    confidence_score = proposals_score

    # Read image
    image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

    # Copy image as original
    org = image.copy()

    # IoU threshold
    threshold = 0.8

    keep_boxes, keep_scores = nms(bounding_boxes, confidence_score, threshold)
