import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN

def createGaussianBPFilter(shape, center, bandCenter, bandWidth):
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.sqrt(np.power(c, 2.0) + np.power(r, 2.0))
    lpFilter_matrix = np.zeros(shape, np.float32)
    lpFilter = np.exp(-pow((d-pow(bandCenter,2))/(d*bandWidth), 2))
    lpFilter_matrix[:, :, 0] = lpFilter
    lpFilter_matrix[:, :, 1] = lpFilter
    return lpFilter_matrix

# def spot_droplet_via_fft(image):
#     dft = np.fft.fft2(image)
#     dft_shift = np.fft.fftshift(dft)
    
#     rows, cols = image.shape
#     num = 2 # determine how much information with low frequency will be dropped
#     crow,ccol = rows//2 , cols//2
#     # num = 150
#     # dft_shift[crow-num:crow+num, ccol-num:ccol+num] = 0.1

#     gaussianBPFilter = createGaussianBPFilter(image.shape, 150, 200)
#     dft_shift = gaussianBPFilter * dft_shift
    
#     f_ishift = np.fft.ifftshift(dft_shift)
#     img_back = np.fft.ifft2(f_ishift)
#     img_back = img_back + np.min(img_back)
#     return img_back

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
    bw_image  = cv.threshold(image,np.sort(image.flatten())[-300], 255, cv.THRESH_BINARY)[1]
    return bw_image

def add_contrast(image):
    square_image = np.square(image.astype('float32'))
    max_square_image = np.max(square_image)
    square_image = np.round((square_image / max_square_image * image))
    return square_image

def remove_dawn_pixels(image, threshold = 10000):
    for row_index, row in enumerate(image):
        for col_index, col in enumerate(row):
            if col < threshold:
                row[col_index] = 0


def stdFftImage(img_gray, rows, cols):
    fimg = np.copy(img_gray)
    fimg = fimg.astype(np.float32)   
    for r in range(rows):
        for c in range(cols):
            if (r+c) % 2:
                fimg[r][c] = -1 * img_gray[r][c]
    img_fft = fftImage(fimg, rows, cols)
    return img_fft

def fftImage(img_gray, rows, cols):
    rPadded = cv.getOptimalDFTSize(rows)
    cPadded = cv.getOptimalDFTSize(cols)
    imgPadded = np.zeros((rPadded, cPadded), dtype=np.float32)
    imgPadded[:rows, :cols] = img_gray
    img_fft = cv.dft(imgPadded, flags=cv.DFT_COMPLEX_OUTPUT)
    return img_fft

def graySpectrum(fft_img):
    real = np.power(fft_img[:, :, 0], 2.0)
    imaginary = np.power(fft_img[:, :, 1], 2.0)
    amplitude = np.sqrt(real+imaginary)
    spectrum = np.log(amplitude+1.0)
    spectrum = cv.normalize(spectrum, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    spectrum *= 255
    return amplitude, spectrum

def spot_droplet_via_fft(image):
    image = ( np.copy(image) / 256 ).astype('uint8')
    rows, cols = image.shape[:2]
    img_fft = stdFftImage(image, rows, cols)
    amplitude, _ = graySpectrum(img_fft)
    minValue, maxValue, minLoc, maxLoc = cv.minMaxLoc(amplitude)
    nrows, ncols = img_fft.shape[:2]
    ilpFilter = createGaussianBPFilter(img_fft.shape, maxLoc, 10, 100)
    img_filter = ilpFilter * img_fft
    img_ift = cv.dft(img_filter, flags=cv.DFT_INVERSE + cv.DFT_REAL_OUTPUT + cv.DFT_SCALE)
    ori_img = np.copy(img_ift[:rows, :cols])
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2:
                ori_img[r][c] = -1 * ori_img[r][c]
            if ori_img[r][c] < 0:
                ori_img[r][c] = 0
            if ori_img[r][c] > 255:
                ori_img[r][c] = 255


    return ori_img