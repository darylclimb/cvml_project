"""
A script to demonstrate geometric transformation (similarity).
                x' = [sR  t] * x
Rotation about image center
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Parameters
angle = 60
scale = 1.5
tx, ty = 26, 10

image_path = 'image.jpg'
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

# image center
height, width = image.shape[:2]
cx, cy = np.array((width // 2, height // 2)).astype(float)


def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))).astype(int) if homogenous else coords


def inverse_warped():
    """
    Inverse warp with nearest neighbour interpolation
    """
    R = np.array([
        [np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0],
        [-np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
        [0, 0, 1]
    ])

    T = np.array([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]
    ])

    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])
    # Transform matrix
    transform_mat = T @ R @ S @ np.linalg.inv(T)
    transform_mat[0, 2] += tx
    transform_mat[1, 2] += ty

    transform_mat_inv = np.linalg.inv(transform_mat)

    # Apply inverse transform and round it (nearest neighbour interpolation)
    coords = get_grid(width, height, True)
    x2, y2 = coords[0], coords[1]
    warp_coords = np.round(transform_mat_inv @ coords).astype(int)
    x1, y1 = warp_coords[0, :], warp_coords[1, :]

    # Get pixels within image boundaries
    indices = np.where((x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height))

    # Map Correspondence
    xpix1, ypix1 = x2[indices], y2[indices]
    xpix2, ypix2 = x1[indices], y1[indices]
    warped = np.zeros_like(image)
    warped[ypix1, xpix1] = image[ypix2, xpix2]

    return warped


def transform_opencv():
    # Transform matrix
    transform_mat = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    transform_mat[0, 2] += tx
    transform_mat[1, 2] += ty

    # Apply transformation
    warped = cv2.warpAffine(image, transform_mat, (width, height), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
    return warped


if __name__ == '__main__':
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f'Angle: {angle}, Scale: {scale}, Translate: tx {tx}, ty {ty}', fontsize=12)

    ax[0].imshow(image)
    ax[0].set_title('Image')

    # inverse warp
    warped = inverse_warped()
    ax[1].imshow(warped)
    ax[1].set_title('Transform')

    # opencv transformation
    warped_opencv = transform_opencv()
    ax[2].imshow(warped_opencv)
    ax[2].set_title('OpenCV Transform')

    plt.tight_layout()
    plt.show()
