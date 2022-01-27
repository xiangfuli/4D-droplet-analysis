import skvideo.io
import numpy as np

def output_analysis_video(path, original_images, images_after_preprocessed, droplet_spotting_images, channel_fused=False):
    # check if all images have the same shape
    assert original_images.shape == images_after_preprocessed.shape and original_images.shape == droplet_spotting_images.shape
   
    output_images = np.append(
        np.append(original_images, images_after_preprocessed, axis = 3),
        droplet_spotting_images, axis = 3)

    channel_num, time_series_num, height, width = output_images.shape
    output_images = np.swapaxes(output_images, 0, 1)
    output_images = output_images.reshape((time_series_num, channel_num * height, width))

    # seems like skvideo can't handle 16bit depth images, we have to do some changes here


    skvideo.io.vwrite(path, output_images, inputdict={'-r':str(1), "-pix_fmt": "gray"})
