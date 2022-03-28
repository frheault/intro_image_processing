#!/usr/bin/env python

""" Remove background of image using pixel value from a position"""

import argparse
import os

import imageio
import numpy as np
from skimage.segmentation import flood_fill



def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_filename',
                   help='Path of the input image.')
    p.add_argument('out_filename',
                   help='Path of the output image.')
    p.add_argument('--mode', choices=['fill', 'value'], default='fill',
                   help='Either replace all close values or use a floodfill. [%(default)s]')
    p.add_argument('--position', nargs=2, type=int, default=[0, 0],
                   help='Position at which the white vs black background is detected. [%(default)s]')
    p.add_argument('--threshold', type=float, default=10,
                   help='Threshold for background detection in case of noise. [%(default)s]')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    args.position = tuple(args.position)

    image = imageio.imread(args.in_filename)
    shape = image.shape[0:2] + (4,)
    tmp = np.average(image, axis=-1)

    # Black background vs White background
    if tmp[args.position] < args.threshold:
        tmp[tmp < args.threshold] = 0
    elif tmp[args.position] > 255 - args.threshold:
        tmp[tmp > 255 - args.threshold] = 255

    if args.mode == 'fill':
        binary_struct = np.zeros((3,3))
        binary_struct[0:3, 1] = 1
        binary_struct[1, 0:3] = 1

        filled_image = flood_fill(tmp, args.position, 1000,
                                  footprint=binary_struct, tolerance=1)
        filled_image[filled_image != 1000] = 255
        filled_image[filled_image == 1000] = 0
    else:
        filled_image = np.zeros(shape[0:2])
        filled_image[tmp == tmp[args.threshold]] = -1
        filled_image[tmp != tmp[args.threshold]] = 255
        filled_image[filled_image == -1] = 0

    # The detected background is used in the Alpha channel
    new_image = np.zeros(shape).astype(np.uint8)
    new_image[:,:,0:3] = image[:,:,0:3]
    filled_image = filled_image
    new_image[:,:,3] = filled_image

    imageio.imwrite(args.out_filename, new_image)


if __name__ == "__main__":
    main()
