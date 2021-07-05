import numpy as np
import cv2 as cv
from scipy.spatial.distance import euclidean, cosine
from matplotlib import pyplot as plt

def get_centers(cntrs):
    if isinstance(cntrs, list):
        centers = []

        for cntr in cntrs:
            M = cv.moments(cntr)
            # assert M['m00'] != 0
            if M['m00'] == 0:
                print(cntr)

            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append([cX, cY])
        return np.array(centers)

    else:
        M = cv.moments(cntrs)
        # assert M['m00'] != 0
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers = [cX, cY]
        return np.array(centers)


def find_nearest(array, value):

    array = np.asarray(array)
    distances = np.linalg.norm((array - value), axis=1)

    return np.min(distances), np.argmin(distances)


def find_nearest_to_center_cntr(cntrs, imsize):
    im_center = np.array([imsize[1] // 2, imsize[0] // 2])

    centers = get_centers(cntrs)
    dist, nearest = find_nearest(centers, im_center)

    if dist > 100:
        return Noneboxes

    return cntrs[nearest]


def get_padded_image(img):
    # Getting the bigger side of the image
    s = max(img.shape[0:2])

    # Creating a dark square with NUMPY
    f = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    ax, ay = (s - img.shape[1])//2, (s - img.shape[0])//2

    # Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img

    return f


def get_nearest_to_center_box(im_shape, boxes):
    center = np.array(im_shape[:-1]) // 2
    min_dist = 1000000  # just a big number
    min_idx = -1
    for idx, box in enumerate(boxes):
        box_center = ((box[3] + box[1]) // 2, (box[2] + box[0]) // 2)
        dist = cosine(box_center, center)
        if dist < min_dist:
            min_dist = dist
            min_idx = idx

    return min_idx


def get_one_mask(boxes, mask, image, n_mask=None):
    if n_mask is None:
        cent_ix = get_nearest_to_center_box(image.shape, boxes)
    else:
        cent_ix = n_mask
    x1, y1, x2, y2 = boxes[cent_ix]
    sz = (x2 - x1, y2 - y1)
    mask_rs = cv.resize(mask[cent_ix].squeeze().detach().cpu().numpy(), sz)

    cur_mask = np.zeros((image.shape[: -1]))
    cur_mask[y1: y2, x1: x2] = mask_rs
    ret, res = cv.threshold(cur_mask, 0.5, 1.0, cv.THRESH_BINARY)
    
    
    return res


def removeOutliers(x, outlierConstant):
    # a = np.array(x)
    # print(a.shape)
    cur_x = x.clone()
    # inliers = np.array()
    for col in range(x.shape[1]):
        a = cur_x[:, col]

        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

        cur_x = cur_x[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]
        print(cur_x.shape)

    # print(cur_x)
    return cur_x
    # return np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))
