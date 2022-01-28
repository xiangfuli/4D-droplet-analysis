#!/usr/bin/python
from email.mime import image
from pickletools import uint8
from utils.nd2_reader import read, visualize_fused_images, fuse_in_time_series_by_z_axis, get_metadata
from utils.droplet_utils import spot_droplet_via_fft, group_by, marginalize, remove_dawn_pixels, add_contrast
from utils.video_utils import output_analysis_video
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


if __name__ == "__main__":
    path = "/Users/eric.lee/Downloads/p88 TFH.nd2"
    path = "/Users/eric.lee/Downloads/4d.nd2"
    
    images = read(path)
    images = fuse_in_time_series_by_z_axis(images)

    shape = images.shape

    original_images = np.copy(images)
    # plt.imshow(images[8], cmap='Greys')

    for channel_index, channel_images in enumerate(images):
        for index, image in enumerate(channel_images):
            # channel_images[index] = cv.GaussianBlur(image, (5, 5), 3)
            channel_images[index] = spot_droplet_via_fft(image)
            # time_series_images[index] = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 11)
            # time_series_images[index] = cv.equalizeHist(image)
            # cv.dilate(image, (5, 20), time_series_images[index])
            
            
            # plt.hist(image.flatten())
            # plt.show()           
    images_after_fft = np.copy(images)

    for channel_index, channel_images in enumerate(images):
        for index, image in enumerate(channel_images):
            channel_images[index] = marginalize(image)

    print('start grouping...')
    points_nums = np.zeros((shape[0], shape[1]))
    circled_images = np.zeros(images.shape)
    for channel_index, channel_images in enumerate(images):
        for index, image in enumerate(channel_images):
            points = group_by(image, 1, 1)
            # points = cv.HoughCircles(np.uint8(frame), cv.HOUGH_GRADIENT, 2, 40, param2=8, minRadius=2, maxRadius=20)
            # points = np.uint8(np.round(points))
            print("index: %s, circle numbers: %s" % (index, len(points)))

            for point in points:
                # draw the outer circle
                # cv.circle(gray_img, (circle[0], circle[1]), circle[2], (255, 255, 255), 2)
                cv.circle(circled_images[channel_index][index], (point[1], point[0]), 1, 255, cv.FILLED, 2)
                # cv.circle(circled_images[index], (point[0], point[1]), point[2], 255, cv.FILLED, 2)
            points_nums[channel_index, index] = len(points)
    # visualize_fused_images(original_image, images, circled_images)

    original_image = ( np.copy(original_images) / 256 ).astype('uint8')
    # images = ( np.copy(images) / 256 ).astype('uint8')

    print('genereating video...')
    output_analysis_video(path.replace(".nd2", ".mp4"), original_image, images_after_fft, images, circled_images)

    # for i in range(shape[0]):
    #     x = np.arange(0, shape[1])
    #     plt.plot(x, points_nums[i], label='channel %s' + str(i))

    # x = np.arange(0, shape[1])
    # plt.plot(x, [58,76,81,76,74,78,75,70,73,69,72,72,71,64,68,72,67,69,68,64,65,70,67,68,67,66,64,62,67,65,61,66,60,65,63,60,59,61,56,59,57,54,53,60,56,53,54,51,54,54,49,53,50,53,47,51,49,53,48,50], label='true num in channel %s' + str(i))
    # plt.plot(x, [0,52,62,67,64,64,68,63,69,61,68,60,62,63,63,60,61,57,56,55,58,61,61,57,53,52,53,55,49,52,51,47,51,44,48,45,43,44,45,45,46,43,41,43,41,40,41,42,42,40,40,36,38,36,36,35,34,34,33,33], label='true num in channel %s' + str(i))

    # plt.legend()
    # plt.show()
