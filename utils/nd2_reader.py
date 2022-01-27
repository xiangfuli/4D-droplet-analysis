#!/usr/bin/python

from math import floor
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import colors
from pims import ND2_Reader


def read(file_path):
    frames = ND2_Reader(file_path)

    assert len(frames) != 0

    # first get the channel info
    meta = frames.metadata
    channelNum = meta['plane_count']
    x = meta['width']
    y = meta['height']
    t = len(frames)
    z = len(frames[0])
    images = np.zeros(shape=(channelNum, t, z, x, y))

    for channelIndex in range(channelNum):
        frames.default_coords['c'] = channelIndex
        images[channelIndex] = np.copy(frames)

    return images

def get_metadata(file_path):
    return ND2_Reader(file_path).metadata

def visualize_original_images(meta, images):
    shape = images.shape
    for channel in range(shape[0]):
        fig, axes = plt.subplots(nrows=shape[1], ncols=shape[2], figsize=(50,50)) # rows woth time, columns with z
        for i, ax in enumerate(axes.flat, start=0):
            if i + 1 <= shape[2]:
                ax.set_title('Z={}'.format(i + 1))
            ax.imshow(images[channel, int(i / shape[2]), i % shape[2]], cmap='Greys')
    plt.show()

def visualize_fused_images(original_image, image_after_preprocess, images):
    shape = images.shape
    fig, axes = plt.subplots(nrows=shape[0], ncols=3, figsize=(50,50)) # rows woth time, columns with z
    for i, ax in enumerate(axes.flat, start=0):
        ax.set_title('T={}'.format(floor(i/3) + 1))
        if i % 3 == 0:
            ax.imshow(original_image[int(i / 3)], cmap='Greys')
        elif i % 3 == 1:
            ax.imshow(image_after_preprocess[int(i / 3)], cmap='Greys')
        else:
            ax.imshow(images[int(i / 3)], cmap='Greys')
    plt.show()

def fuse_in_time_series_by_z_axis(images):
    shape = list(images.shape)
    shape[2] = 1
    fused = np.zeros(shape=shape)
    for channel in range(shape[0]):
        for timeIndex in range(shape[1]):
            fused[channel, timeIndex] = np.amax(images[channel, timeIndex], axis=0)
    fused.resize((shape[0], shape[1], shape[3], shape[4]))
    return fused
