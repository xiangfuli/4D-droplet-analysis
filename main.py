#!/usr/bin/python
from email.mime import image
from pickletools import uint8
from utils.nd2_reader import read, visualize_fused_images, fuse_in_time_series_by_z_axis, get_metadata
from utils.droplet_utils import spot_droplet_via_fft, group_by, marginalize
from utils.video_utils import output_analysis_video
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


if __name__ == "__main__":
    images = read("/Users/eric.lee/Downloads/p88 TFH.nd2")
    images = fuse_in_time_series_by_z_axis(images)

    original_image = ( np.copy(images) / 256 ).astype('uint8')
    # plt.imshow(images[8], cmap='Greys')

    for time_series_images in images:
        for index, image in enumerate(time_series_images):
            time_series_images[index] = spot_droplet_via_fft(image)
            time_series_images[index] = marginalize(image)
    
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
                cv.circle(circled_images[channel_index][index], (point[1], point[0]), 3, 255, cv.FILLED, 2)
                # cv.circle(circled_images[index], (point[0], point[1]), point[2], 255, cv.FILLED, 2)

    # visualize_fused_images(original_image, images, circled_images)
    output_analysis_video("/Users/eric.lee/Downloads/video.mp4", original_image, images, circled_images)