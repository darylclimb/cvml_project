import json

import numpy as np


# =========================================================
# Geometry
# =========================================================
def rotation_from_euler(roll=1., pitch=1., yaw=1.):
    """
    Get rotation matrix
    Args:
        roll, pitch, yaw:       In radians

    Returns:
        R:          [4, 4]
    """
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(4)
    R[0, 0] = cj * ck
    R[0, 1] = sj * sc - cs
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci
    return R


def translation_matrix(vector):
    """
    Translation matrix

    Args:
        vector list[float]:     (x, y, z)

    Returns:
        T:      [4, 4]
    """
    M = np.identity(4)
    M[:3, 3] = vector[:3]
    return M


def load_camera_params(file):
    """
    Get the intrinsic and extrinsic parameters
    Returns:
        Camera extrinsic and intrinsic matrices
    """
    with open(file, 'rt') as handle:
        p = json.load(handle)

    fx, fy = p['fx'], p['fy']
    u0, v0 = p['u0'], p['v0']

    pitch, roll, yaw = p['pitch'], p['roll'], p['yaw']
    x, y, z = p['x'], p['y'], p['z']

    # Intrinsic
    K = np.array([[fx, 0, u0, 0],
                  [0, fy, v0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    # Extrinsic
    R_veh2cam = np.transpose(rotation_from_euler(roll, pitch, yaw))
    T_veh2cam = translation_matrix((-x, -y, -z))

    # Rotate to camera coodinates
    R = np.transpose(np.array([[0., 0., 1., 0.],
                               [1., 0., 0., 0.],
                               [0., -1., 0., 0.],
                               [0., 0., 0., 1.]]))

    RT = R @ R_veh2cam @ T_veh2cam
    return RT, K


# =========================================================
# Projections
# =========================================================
def perspective(cam_coords, proj_mat, h, w):
    """
    P = proj_mat @ (x, y, z, 1)
    Project cam2pixel

    Args:
        cam_coords:         [4, npoints]
        proj_mat:           [4, 4]

    Returns:
        pix coords:         [h, w, 2]
    """
    eps = 1e-7
    pix_coords = proj_mat @ cam_coords

    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + eps)
    pix_coords = np.reshape(pix_coords, (2, h, w))
    pix_coords = np.transpose(pix_coords, (1, 2, 0))
    return pix_coords


def bilinear_sampler(imgs, pix_coords):
    """
    Construct a new image by bilinear sampling from the input image.
    Args:
        imgs:                   [H, W, C]
        pix_coords:             [h, w, 2]
    :return:
        sampled image           [h, w, c]
    """
    img_h, img_w, img_c = imgs.shape
    pix_h, pix_w, pix_c = pix_coords.shape
    out_shape = (pix_h, pix_w, img_c)

    pix_x, pix_y = np.split(pix_coords, [1], axis=-1)  # [pix_h, pix_w, 1]
    pix_x = pix_x.astype(np.float32)
    pix_y = pix_y.astype(np.float32)

    # Rounding
    pix_x0 = np.floor(pix_x)
    pix_x1 = pix_x0 + 1
    pix_y0 = np.floor(pix_y)
    pix_y1 = pix_y0 + 1

    # Clip within image boundary
    y_max = (img_h - 1)
    x_max = (img_w - 1)
    zero = np.zeros([1])

    pix_x0 = np.clip(pix_x0, zero, x_max)
    pix_y0 = np.clip(pix_y0, zero, y_max)
    pix_x1 = np.clip(pix_x1, zero, x_max)
    pix_y1 = np.clip(pix_y1, zero, y_max)

    # Weights [pix_h, pix_w, 1]
    wt_x0 = pix_x1 - pix_x
    wt_x1 = pix_x - pix_x0
    wt_y0 = pix_y1 - pix_y
    wt_y1 = pix_y - pix_y0

    # indices in the image to sample from
    dim = img_w

    # Apply the lower and upper bound pix coord
    base_y0 = pix_y0 * dim
    base_y1 = pix_y1 * dim

    # 4 corner vertices
    idx00 = (pix_x0 + base_y0).flatten().astype(np.int)
    idx01 = (pix_x0 + base_y1).astype(np.int)
    idx10 = (pix_x1 + base_y0).astype(np.int)
    idx11 = (pix_x1 + base_y1).astype(np.int)

    # Gather pixels from image using vertices
    imgs_flat = imgs.reshape([-1, img_c]).astype(np.float32)
    im00 = imgs_flat[idx00].reshape(out_shape)
    im01 = imgs_flat[idx01].reshape(out_shape)
    im10 = imgs_flat[idx10].reshape(out_shape)
    im11 = imgs_flat[idx11].reshape(out_shape)

    # Apply weights [pix_h, pix_w, 1]
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1
    output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11
    return output


class Plane:
    """
    Defines a plane in the world
    """

    def __init__(self, x, y, z, roll, pitch, yaw,
                 col, row, scale):
        self.x, self.y, self.z = x, y, z
        self.roll, self.pitch, self.yaw = roll, pitch, yaw

        self.col, self.row = col, row
        self.scale = scale

        self.xyz = self.xyz_coord()

    def xyz_coord(self):
        """
        Returns:
            Grid coordinate: [b, 3/4, row*cols]
        """
        xmin = self.x
        xmax = self.x + self.col * self.scale
        ymin = self.y
        ymax = self.y + self.row * self.scale
        return meshgrid(xmin, xmax, self.col,
                        ymin, ymax, self.row)


def meshgrid(xmin, xmax, num_x, ymin, ymax, num_y, is_homogeneous=True):
    """
    Grid is parallel to z-axis

    Returns:
        array x,y,z,[1] coordinate   [3/4, num_x * num_y]
    """
    x = np.linspace(xmin, xmax, num_x)
    y = np.linspace(ymin, ymax, num_y)
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)

    if is_homogeneous:
        coords = np.stack([x, y, z, np.ones_like(x)], axis=0)
    else:
        coords = np.stack([x, y, z], axis=0)
    return coords
