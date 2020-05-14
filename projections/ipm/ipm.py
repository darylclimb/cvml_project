import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import perspective, Plane, load_camera_params, bilinear_sampler

image = cv2.cvtColor(cv2.imread('stuttgart_01_000000_003715_leftImg8bit.png'), cv2.COLOR_BGR2RGB)
TARGET_H, TARGET_W = 500, 500


def ipm_from_parameters(image, xyz, K, RT):
    P = K @ RT
    pixel_coords = perspective(xyz, P, TARGET_H, TARGET_W)
    image2 = bilinear_sampler(image, pixel_coords)
    return image2.astype(np.uint8)


def ipm_from_opencv(image, source_points, target_points):
    # Compute projection matrix
    M = cv2.getPerspectiveTransform(source_points, target_points)
    # Warp the image
    warped = cv2.warpPerspective(image, M, (TARGET_W, TARGET_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    return warped

if __name__ == '__main__':
    ################
    # Derived method
    ################
    # Define the plane on the region of interest (road)
    plane = Plane(0, -25, 0, 0, 0, 0, TARGET_H, TARGET_W, 0.1)
    # Retrieve camera parameters
    extrinsic, intrinsic = load_camera_params('camera.json')
    # Apply perspective transformation
    warped1 = ipm_from_parameters(image, plane.xyz, intrinsic, extrinsic)

    ################
    # OpenCV
    ################
    # Vertices coordinates in the source image
    s = np.array([[830, 598],
                  [868, 568],
                  [1285, 598],
                  [1248, 567]], dtype=np.float32)

    # Vertices coordinates in the destination image
    t = np.array([[177, 231],
                  [213, 231],
                  [178, 264],
                  [216, 264]], dtype=np.float32)

    # Warp the image
    warped2 = ipm_from_opencv(image, s, t)

    # Draw results
    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(image)
    ax[0].set_title('Front View')
    ax[1].imshow(warped1)
    ax[1].set_title('IPM')
    ax[2].imshow(warped2)
    ax[2].set_title('IPM from OpenCv')

    plt.show()
