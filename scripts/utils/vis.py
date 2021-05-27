import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



def visualize_segmentation(im, masks, nc=None, return_rgb=False, save_dir=None):

    """ Visualize segmentations nicely. Based on code from:
        https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

        @param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
        @param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., K}
        @param nc: total number of colors. If None, this will be inferred by masks
    """ 
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    im = im.copy()

    if not return_rgb:
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(im)

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    if not return_rgb:
        # matplotlib stuff
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.axis('off')
        fig.add_axes(ax)

    # Mask
    imgMask = np.zeros(im.shape)


    # Draw color masks
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)


        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)


    # Draw mask contours
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Find contours
        try:
            contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        except:
            im2, contour, hier = cv2.findContours(
                e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Plot the nice outline
        for c in contour:
            if save_dir is None and not return_rgb:
                polygon = Polygon(c.reshape((-1, 2)), fill=False, facecolor=color_mask, edgecolor='w', linewidth=1.2, alpha=0.5)
                ax.add_patch(polygon)
            else:
                cv2.drawContours(im, contour, -1, (255,255,255), 2)


    if save_dir is None and not return_rgb:
        ax.imshow(im)
        return fig
    elif return_rgb:
        return im
    elif save_dir is not None:
        # Save the image
        PIL_image = Image.fromarray(im)
        PIL_image.save(save_dir)
        return PIL_image


def get_rects(mask):
    # print(np.unique(mask))
    # plt.imshow(mask)
    # plt.show()
    # Detect edges using Canny
    # contours = []
    # for cl in np.unique(mask)[1:]:
        
    #     # one_cl_mask = mask[np.where(mask == cl)]
    #     # mask = cv2.inRange(mask, cl, cl)
    #     one_cl_mask = mask.copy()
    #     one_cl_mask[mask != cl] = 0
    #     one_cl_mask[mask == cl] = 255
    #     # print('class:',cl)
    #     # print(np.unique(one_cl_mask))

    #     kernel = (12, 12)
    #     one_cl_mask = cv.erode(one_cl_mask, kernel)
    #     one_cl_mask = cv.dilate(one_cl_mask, kernel)

    #     canny_output1= cv.Canny(one_cl_mask, 0, 1)

    #     contours1, _ = cv.findContours(canny_output1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #     cv.drawContours(one_cl_mask, contours1, -1, 128, 2)

    #     contours += contours1

        # plt.imshow(one_cl_mask)
        # plt.show()

    # canny_output = cv.Canny(mask, 0, 1)
    # Find contours
    # contours, hierarchy = cv.findContours(canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print(hierarchy)
    # print()
    # print(contours)

    outer_cntrs = []

    # cv.drawContours(im, contours, -1, (255, 0, 0), 2)
    # plt.imshow(im)
    # plt.show()
    # print(type(contours))
    # print('contours:', len(contours))
    big_cntrs = []
    small_cntrs = []
    for idx, cntr in enumerate(contours):

        rect = cv.minAreaRect(cntr)

        area = rect[1][0]*rect[1][1]

        # print(area, cv.contourArea(cntr))

        c = mask.copy()
        cv.drawContours(c, [cntr], -1, 10, 2)

        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(c,[box],0,7,2)

        # plt.imshow(c)
        # plt.show()

        

        # if cv.contourArea(cntr) > 50: #and (hierarchy[0][idx][3] != -1):
        if area > 1000: #and (hierarchy[0][idx][3] != -1):
            
            # print(contours[contours==cntr])
            big_cntrs.append(cntr)
            # contours.remove(cntr)
            
        else:
            small_cntrs.append(cntr)
            # outer_cntrs.append(cntr)

    contours = big_cntrs
    # print('big contnours', len(big_cntrs))
    rects = []

    for cntr in contours:
        rect = cv.minAreaRect(cntr) # basically you can feed this rect into your classifier
        (x,y),(w,h), a = rect # a - angle
        rects.append(rect)

    return rects, contours, small_cntrs


def get_rotated_rois(rgb_im, depth_im, mask):
    rects, cntrs,_ = get_rects(mask)

    warped_rgb = []
    warped_depth = []
    for rect in rects:
        warped_rgb.append(crop_rect(rgb_im, rect))
        warped_depth.append(crop_rect(depth_im, rect))
        # warped.append((warped_rgb, warped_depth))
 
    return warped_rgb, warped_depth, cntrs


def crop_rect(img, rect):

    box = cv.boxPoints(rect)
    box = np.int0(box) #turn into ints    
    # get width and height of the detected rectangle
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
  
    return warped


# def find_nearest_prev_rois(prev_mask, mask):
