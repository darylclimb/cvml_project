import os
from collections import Counter
import matplotlib as mpl
import numpy as np
import tensorflow as tf
from matplotlib import cm as cm


###########################################################################
# Projection utils
###########################################################################
def bilinear_sampler(imgs, coords):
    """
    Construct a new image by bilinear sampling from the input image.
    Args:
        imgs: source image to be sampled from [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, height_t,

    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """

    def _repeat(x, n_repeats):
        rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')
    eps = tf.constant([0.5], tf.float32)

    coords_x = tf.clip_by_value(coords_x, eps, x_max - eps)
    coords_y = tf.clip_by_value(coords_y, eps, y_max - eps)

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    # indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(_repeat(tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                              coord_size[1] * coord_size[2]),
                      [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output


def pixel2cam(depth, pixel_coords, intrinsic_mat_inv, homogenous=True):
    """
    Transform coordinates in the pixel frame to the camera frame using depth.
    inv(K) @ (u, v) * depth
    Args:
        depth:              [b, 1, npoints]
        pixel_coords:       [b, 3, npoints
        intrinsic_mat_inv:  [b, 4, 4]
    Returns:
        cam_coords:         [b, 3/4, npoints]
    """
    cam_coords = tf.matmul(intrinsic_mat_inv[:, :3, :3], pixel_coords) * depth
    if homogenous:
        cam_coords = tf.concat((cam_coords, tf.ones_like(depth)), axis=1)
    return cam_coords


def backproject(pixel_coords, depth, intrinsics_inv, homogenous=True):
    """
    Args:
        pixel_coords:       [b, 3, npixels]
        depth:              [b, h, w]
        intrinsics_inv:     [b, 4, 4]
    Returns:
        cam_coords: [b, 4, npoints] homogenous coordinate

    """
    dims = tf.shape(depth)
    batch_size, img_height, img_width = dims[0], dims[1], dims[2]
    depth = tf.reshape(depth, [batch_size, 1, img_height * img_width])

    # Apply transform
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics_inv, homogenous)  # [b, 3, npoints]
    return cam_coords


def forwardproject(cam_coords, proj_mat, h, w):
    """
    P = proj_mat @ (x, y, z, 1)
    Project cam2pixel coordinate

    Args:
        cam_coords:         [b, 4, npoints]
        proj_mat:           [b, 4, 4]

    Returns:
        pix coords:         [b, h, w, 2]
    """
    batch_size = tf.shape(cam_coords)[0]
    cam_points = tf.matmul(proj_mat, cam_coords)

    eps = 1e-7
    pix_coords = cam_points[:, :2, :] / (tf.expand_dims(cam_points[:, 2, :], 1) + eps)  # [b, 2, npoints]
    pix_coords = tf.reshape(pix_coords, (batch_size, 2, h, w))  # [b, 2, h, w]
    pix_coords = tf.transpose(pix_coords, (0, 2, 3, 1))  # [b, h, w, 2]
    return pix_coords


###########################################################################
# Training utils
###########################################################################
def ssim_loss(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = tf.nn.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = tf.nn.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x = tf.nn.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y = tf.nn.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)


def gradient_x(img):
    paddings = tf.constant([[0, 0], [0, 0], [1, 0], [0, 0]])
    img = tf.pad(img, paddings, 'CONSTANT')
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx


def gradient_y(img):
    paddings = tf.constant([[0, 0], [1, 0], [0, 0], [0, 0]])
    img = tf.pad(img, paddings, 'CONSTANT')
    gy = img[:, :-1, :, :] - img[:, 1:, :, :]
    return gy


def smooth_loss(disp, img):
    """
    Compute L1 penalty by weighting image gradient
    :param tensor disp:
    :param list tensor pyramid:
    :return:
    """
    disp_gradients_x = tf.abs(gradient_x(disp))
    disp_gradients_y = tf.abs(gradient_y(disp))

    image_gradients_x = tf.abs(gradient_x(img))
    image_gradients_y = tf.abs(gradient_y(img))

    weights_x = tf.exp(-tf.reduce_mean(image_gradients_x, 3, keepdims=True))
    weights_y = tf.exp(-tf.reduce_mean(image_gradients_y, 3, keepdims=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y
    return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)


def pixel_coord(batch_size, height, width, homogenous=True):
    """
    Create pixel coordinates. Meshgrid in the absolute coordinates.

    Returns:
        grid:         [2/3, height*width]
    """
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))

    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))

    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)

    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    if homogenous:
        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], axis=0)
    else:
        grid = tf.concat([x_t_flat, y_t_flat], axis=0)
    grid = tf.tile(tf.expand_dims(grid, 0), [batch_size, 1, 1])

    return grid


###########################################################################
# Eval utils
###########################################################################
def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1. / max_depth
    max_disp = 1. / min_depth
    scaled_disp = tf.cast(min_disp, tf.float32) + tf.cast((max_disp - min_disp), tf.float32) * disp
    depth = 1. / scaled_disp
    return scaled_disp, depth


def compute_errors(gt, pred):
    """
    Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """
    Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth


###########################################################################
# Visualisation
###########################################################################
def visualize_colormap(mat):
    # high_res_colormap
    vmax = np.percentile(mat, 95)
    normalizer = mpl.colors.Normalize(vmin=mat.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(mat)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


###########################################################################
# OS
###########################################################################
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
