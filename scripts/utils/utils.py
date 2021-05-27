import numpy as np
import cv2 as cv




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
        return None

    return cntrs[nearest]


    