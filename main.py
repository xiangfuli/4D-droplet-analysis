#!/usr/bin/python
from email.mime import image
from pickletools import uint8
from utils.nd2_reader import read, visualize_fused_images, fuse_in_time_series_by_channel_and_z_axis, get_metadata
from utils.droplet_utils import spot_droplet_via_fft, gooup_by, marginalize
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


if __name__ == "__main__":
    images = read("/Users/eric.lee/Downloads/4d.nd2")
    images = fuse_in_time_series_by_channel_and_z_axis(images)

    original_image = np.copy(images)
    # plt.imshow(images[8], cmap='Greys')

    for index, frame in enumerate(images):
        images[index] = spot_droplet_via_fft(frame)
        images[index] = marginalize(frame)
    
    circled_images = np.zeros(images.shape)
    for index, frame in enumerate(images):
        points = gooup_by(frame, 3, 3)
        # points = cv.HoughCircles(np.uint8(frame), cv.HOUGH_GRADIENT, 2, 40, param2=8, minRadius=2, maxRadius=20)
        # points = np.uint8(np.round(points))
        print("index: %s, circle numbers: %s" % (index, len(points)))

        for point in points:
            # draw the outer circle
            # cv.circle(gray_img, (circle[0], circle[1]), circle[2], (255, 255, 255), 2)
            cv.circle(circled_images[index], (point[1], point[0]), 3, 255, cv.FILLED, 2)
            # cv.circle(circled_images[index], (point[0], point[1]), point[2], 255, cv.FILLED, 2)

    visualize_fused_images(original_image, images, circled_images)