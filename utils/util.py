import os
import time
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import random
import torch
import logging
from scipy import linalg
import imageio
from PIL import Image as pil
from skimage import io, color
from skimage.color import lab2xyz
from skimage.color.colorconv import _convert, _prepare_colorarray, get_xyz_coords

####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)


####################
# image convert
####################


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1), color_type='RGB'):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu()  # clamp
    # tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    if color_type == 'ycbcr_matlab':
        tensor[0, :, :] = tensor[0, :, :].clamp_(16 / 255, 235 / 255)
        tensor[1, :, :] = tensor[1, :, :].clamp_(16 / 255, 240 / 255)
        tensor[2, :, :] = tensor[2, :, :].clamp_(16 / 255, 240 / 255)
    elif color_type == 'lab':
        tensor[0, :, :] = tensor[0, :, :].clamp_(0, 1)
        # tensor[0, :, :] = tensor[0, :, :].clamp_(0, 100)
        tensor[1, :, :] = tensor[1, :, :].clamp_(-1, 1)
        # tensor[1, :, :] = tensor[1, :, :].clamp_(-127, 127)
        tensor[2, :, :] = tensor[2, :, :].clamp_(-1, 1)
        # tensor[2, :, :] = tensor[2, :, :].clamp_(-127, 127)
    elif color_type == 'lab_hdr':
        tensor[1, :, :] = tensor[1, :, :].clamp_(-1, 1)
        # tensor[1, :, :] = tensor[1, :, :].clamp_(-127, 127)
        tensor[2, :, :] = tensor[2, :, :].clamp_(-1, 1)
        # tensor[2, :, :] = tensor[2, :, :].clamp_(-127, 127)
    else:
        tensor = (tensor.clamp_(*min_max) - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        if color_type == 'RGB':
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        elif color_type == 'ycbcr':
            img_np = yuv2rgb_convention(np.transpose(img_np, (1, 2, 0)))[:, :, [2, 1, 0]] # HWC, BGR
        elif color_type == 'ycbcr_matlab':
            img_np = ycbcr2rgb(np.transpose(img_np, (1, 2, 0)))[:, :, [2, 1, 0]] # HWC, BGR
        elif color_type == 'lab':
            img_np = lab2rgb(np.transpose(img_np, (1, 2, 0)))[:, :, [2, 1, 0]] # HWC, BGR
        elif color_type == 'lab_hdr':
            img_np = lab2rgb_hdr(np.transpose(img_np, (1, 2, 0)))[:, :, [2, 1, 0]] # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def tensor2img2(tensor, out_type=np.uint8, min_max=(0, 1), color_type='RGB'):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    # tensor = tensor.squeeze().float().cpu()  # clamp
    # tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    if color_type == 'ycbcr_matlab':
        tensor[0, :, :] = tensor[0, :, :].clip(16 / 255, 235 / 255)
        tensor[1, :, :] = tensor[1, :, :].clip(16 / 255, 240 / 255)
        tensor[2, :, :] = tensor[2, :, :].clip(16 / 255, 240 / 255)
    elif color_type == 'lab':
        tensor[0, :, :] = tensor[0, :, :].clip(0, 1)
        # tensor[0, :, :] = tensor[0, :, :].clamp_(0, 100)
        tensor[1, :, :] = tensor[1, :, :].clip(-1, 1)
        # tensor[1, :, :] = tensor[1, :, :].clamp_(-127, 127)
        tensor[2, :, :] = tensor[2, :, :].clip(-1, 1)
        # tensor[2, :, :] = tensor[2, :, :].clamp_(-127, 127)
    elif color_type == 'lab_hdr':
        tensor[1, :, :] = tensor[1, :, :].clip(-1, 1)
        # tensor[1, :, :] = tensor[1, :, :].clamp_(-127, 127)
        tensor[2, :, :] = tensor[2, :, :].clip(-1, 1)
        # tensor[2, :, :] = tensor[2, :, :].clamp_(-127, 127)
    else:
        tensor = (tensor.clip(*min_max) - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

    n_dim = len(tensor.shape)
    # if n_dim == 4:
    #     n_img = len(tensor)
    #     img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
    #     img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    # elif n_dim == 3:
    assert n_dim == 3
    img_np = tensor
    if color_type == 'RGB':
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif color_type == 'ycbcr':
        img_np = yuv2rgb_convention(np.transpose(img_np, (1, 2, 0)))[:, :, [2, 1, 0]] # HWC, BGR
    elif color_type == 'ycbcr_matlab':
        img_np = ycbcr2rgb(np.transpose(img_np, (1, 2, 0)))[:, :, [2, 1, 0]] # HWC, BGR
    elif color_type == 'lab':
        img_np = lab2rgb(np.transpose(img_np, (1, 2, 0)))[:, :, [2, 1, 0]] # HWC, BGR
    elif color_type == 'lab_hdr':
        img_np = lab2rgb_hdr(np.transpose(img_np, (1, 2, 0)))[:, :, [2, 1, 0]] # HWC, BGR
    # elif n_dim == 2:
    #     img_np = tensor
    # else:
    #     raise TypeError(
    #         'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_imgs_in_batch(imgs, dataset_dir, imgs_path):
    time2 = time.time()
    #time_thread1 = time2 - time1
    for img_iter in range(imgs.shape[0]):
        # print(imgs[img_iter].shape)
        # fds
        sr_img = tensor2img2(imgs[img_iter], color_type='lab')  # uint8
        # sr_img = tensor2img2(imgs[img_iter], color_type='ycbcr_matlab')  # uint8
        # print(imgs[img_iter].shape)
        img_name = os.path.splitext(os.path.basename(imgs_path[img_iter]))[0]
        save_img_path = os.path.join(dataset_dir, img_name + '.png')
        save_img(sr_img, save_img_path)
    # print(time.time()-time2)

def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type).clip(0, 1)


def yuv2rgb_convention(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    # rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
    #                       [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    rlt = np.matmul(img, np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])) + \
          [-179.45477266423404, 135.45870971679688, -226.8183044444304]

    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type).clip(0, 1)

def xyz2rgb_woclip(xyz):
    """XYZ to RGB color space conversion.
    Parameters
    ----------
    xyz : array_like
        The image in XYZ format, in a 3-D array of shape ``(.., .., 3)``.
    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Raises
    ------
    ValueError
        If `xyz` is not a 3-D array of shape ``(.., .., 3)``.
    Notes
    -----
    The CIE XYZ color space is derived from the CIE RGB color space. Note
    however that this function converts to sRGB.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/CIE_1931_color_space
    Examples
    --------
    >>> from skimage import data
    >>> from skimage.color import rgb2xyz, xyz2rgb
    >>> img = data.astronaut()
    >>> img_xyz = rgb2xyz(img)
    >>> img_rgb = xyz2rgb(img_xyz)
    """
    # Follow the algorithm from http://www.easyrgb.com/index.php
    # except we don't multiply/divide by 100 in the conversion
    arr = _convert(rgb_from_xyz, xyz)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
#    np.clip(arr, 0, 1, out=arr)
    return arr


def lab2xyz2(lab, illuminant="D65", observer="2"):
    """CIE-LAB to XYZcolor space conversion.

    Parameters
    ----------
    lab : array_like
        The image in lab format, in a 3-D array of shape ``(.., .., 3)``.
    illuminant : {"A", "D50", "D55", "D65", "D75", "E"}, optional
        The name of the illuminant (the function is NOT case sensitive).
    observer : {"2", "10"}, optional
        The aperture angle of the observer.

    Returns
    -------
    out : ndarray
        The image in XYZ format, in a 3-D array of shape ``(.., .., 3)``.

    Raises
    ------
    ValueError
        If `lab` is not a 3-D array of shape ``(.., .., 3)``.
    ValueError
        If either the illuminant or the observer angle are not supported or
        unknown.
    UserWarning
        If any of the pixels are invalid (Z < 0).


    Notes
    -----
    By default Observer= 2A, Illuminant= D65. CIE XYZ tristimulus values x_ref
    = 95.047, y_ref = 100., z_ref = 108.883. See function 'get_xyz_coords' for
    a list of supported illuminants.

    References
    ----------
    .. [1] http://www.easyrgb.com/index.php?X=MATH&H=07#text7
    .. [2] http://en.wikipedia.org/wiki/Lab_color_space

    """

    arr = _prepare_colorarray(lab).copy()

    L, a, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    y = (L + 16.) / 116.
    x = (a / 500.) + y
    z = y - (b / 200.)

    if np.any(z < 0):
        invalid = np.nonzero(z < 0)
#        warn('Color data out of range: Z < 0 in %s pixels' % invalid[0].size)
        z[invalid] = 0

    out = np.dstack([x, y, z])

    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.)
    out[~mask] = (out[~mask] - 16.0 / 116.) / 7.787

    # rescale to the reference white (illuminant)
    xyz_ref_white = get_xyz_coords(illuminant, observer)
    out *= xyz_ref_white
    return out


# From sRGB specification
xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])


rgb_from_xyz = linalg.inv(xyz_from_rgb)


def lab2rgb_woclip(lab):
    img = xyz2rgb_woclip(lab2xyz2(lab, illuminant="D65", observer="2"))
    return img

def lab2rgb(img):
    '''rgb version of lab2rgb
    only_y: only return L channel
    Input:
        float, L[0, 1], a[-1, 1], b[-1, 1]
    Output:
        float, [0, 1]
    '''
    in_img_type = img.dtype
    # img.astype(np.float32)
    # if in_img_type != np.uint8:
    #     img *= 255.
    assert img.dtype == np.float32
    # rescale to original range(L[0, 100], a[-127, 127], b[-127, 127] and convert
    # L[0, 1], a[-1, 1], b[-1, 1]
    img = img * np.array([100., 127., 127.], dtype=np.float32)
    # L[0, 1], a[0, 1], b[0, 1]
    # img = (img * np.array([100., 255., 255.], dtype=np.float32)) + np.array([0., -128., -128.], dtype=np.float32)

    # rlt = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    # print(img.max(), img.min())
    # fd
    # rlt = color.lab2rgb(img.astype(np.float64))
    rlt = lab2rgb_woclip(img.astype(np.float64))

    # if in_img_type == np.uint8:
    #     print("HDR data should not clip to uint8")
    #     rlt = rlt.round()
    # else:
    #     rlt /= 255.
    return rlt.clip(0, 1).astype(in_img_type)


def lab2rgb_hdr(img):
    '''rgb version of lab2rgb
    only_y: only return L channel
    Input:
        float, L[0, 1], a[-1, 1], b[-1, 1]
    Output:
        float, [0, 1]
    '''
    in_img_type = img.dtype
    # img.astype(np.float32)
    # if in_img_type != np.uint8:
    #     img *= 255.
    assert img.dtype == np.float32
    # rescale to original range(L[0, 100], a[-127, 127], b[-127, 127] and convert
    # L[0, 1], a[-1, 1], b[-1, 1]
    img = img * np.array([100., 127., 127.], dtype=np.float32)
    # L[0, 1], a[0, 1], b[0, 1]
    # img = (img * np.array([100., 255., 255.], dtype=np.float32)) + np.array([0., -128., -128.], dtype=np.float32)

    # rlt = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
    # rlt = color.lab2rgb(img.astype(np.float64))
    rlt = lab2rgb_woclip(img.astype(np.float64))

    # if in_img_type == np.uint8:
    #     print("HDR data should not clip to uint8")
    #     rlt = rlt.round()
    # else:
    #     rlt /= 255.
    return rlt.clip(0, 5).astype(in_img_type)
    # return rlt.clip(0, 1).astype(in_img_type)


def pil_save_image(image, path, space='srgb', icc_path='icc', quality=90):
    '''
    Write (JPEG) image of specified color space.
    :param image:
    :param path:
    :param space:
    :return:
    '''
    if space=='srgb':
        cv2.imwrite(path, image)
        return
    elif space=='709':
        f_icc = os.path.join(icc_path,'Rec709-Rec1886.icc')
    elif space=='2020':
        f_icc = os.path.join(icc_path, 'Rec2020-Rec1886.icc')
    elif space=='dci':
        f_icc = os.path.join(icc_path, 'P3D65.icc')
    elif space=='prophoto':
        f_icc = os.path.join(icc_path, 'ProPhoto.icc')
    else:
        raise ValueError('Invalid color space.')
    try:
        with open(f_icc, 'rb') as f:
            icc = f.read()
    except:
        raise FileNotFoundError('ICC profile not found.')
    if np.amax(image) <= 1:
        image *= 255
    image = image[:, :, [2, 1, 0]]
    pil.fromarray(image.astype('uint8')).save(path, 'jpeg', icc_profile = icc, quality=quality)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def save_img2(img, img_path, mode='RGB'):
    io.imsave(img_path, img[:,:,::-1])


def save_img_hdr(img, img_path):
    img = img[:, :, [2, 1, 0]]
    img = (img.clip(0, 1)*255).astype(np.uint8).astype(np.float32)/255.0
    imageio.imwrite(img_path, img, format='hdr')


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# def cross_entropy(x, y):
#     """ Computes cross entropy between two distributions.
#     Input: x: iterabale of N non-negative values
#            y: iterabale of N non-negative values
#     Returns: scalar
#     """
#
#     if np.any(x < 0) or np.any(y < 0):
#         raise ValueError('Negative values exist.')
#
#     # Force to proper probability mass function.
#     x = np.array(x, dtype=np.float)
#     y = np.array(y, dtype=np.float)
#     x /= np.sum(x)
#     y /= np.sum(y)
#
#     # Ignore zero 'y' elements.
#     mask = y > 0
#     x = x[mask]
#     y = y[mask]
#     ce = -np.sum(x * np.log(y))
#     return ce

def cross_entropy(predictions, targets, epsilon=1e-9):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions))/N
    return ce
