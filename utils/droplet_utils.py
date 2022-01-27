import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def spot_droplet_via_fft(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow,ccol = rows//2 , cols//2
    fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    return img_back

def group_by(thresholding_imgs, x_offset, y_offset):
    groups = []

    for row_index, row in enumerate(thresholding_imgs):
        for col_index, col in enumerate(row):
            if col != 0:
                # print(col)
                # print((row_index, col_index))
                fused = False
                for group_index, group in enumerate(groups):
                    for xy in group:
                        if np.abs(row_index - xy[0]) <= x_offset and np.abs(col_index - xy[1]) <= y_offset:
                            fused=True
                            groups[group_index] += [[row_index, col_index]]
                            break
                    if fused:
                        break
            
                if not fused:
                    groups += [[[row_index, col_index]]]
    
    points_center = []
    for group in groups:
        points_center += [np.uint16(np.around(np.mean(group, axis=0)))]
    return points_center

def marginalize(image):
    # in each image, find the most bright pixels whose value is greater than fix value - 10000
    bw_image  = cv.threshold(image, np.mean(np.sort(image.flatten())[-200:]), 65535, cv.THRESH_BINARY)[1]
    return bw_image