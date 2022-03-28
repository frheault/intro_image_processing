#!/usr/bin/env python

import os
import numpy as np 
import nibabel as nib


axis_enum = {'sagittal':0, 'coronal':1,'axial':2}

def split_name_with_nii(filename):
    base, ext = os.path.splitext(filename)

    if ext == ".gz":
        # Test if we have a .nii additional extension
        tmp_base, extra_ext = os.path.splitext(base)

        if extra_ext == ".nii":
            ext = extra_ext + ext
            base = tmp_base

    return base, ext


def get_slice(volume_3d, axis_name, indice=None):
    # Personal preference for orientation (nose up for axial, nose left for sagittal, nose first for coronal)
    # Operation done in 2D, still very fast and easier to understand
    if axis_name == 'sagittal':
        if indice is None:
            indice = volume_3d.shape[0] // 2
        slice_2d = np.rot90(volume_3d[indice,:,:])
    elif axis_name == 'coronal':
        if indice is None:
            indice = volume_3d.shape[1] // 2
        slice_2d = np.rot90(np.flip(volume_3d, axis=1)[:,indice,:])
    elif axis_name == 'axial':
        if indice is None:
            indice = volume_3d.shape[2] // 2
        slice_2d = np.rot90(np.flip(volume_3d, axis=2)[:,:,indice])
    else:
        raise ValueError('{0} is not a valid axis name'.format(axis))

    return slice_2d


def get_nifti_data(img):
    return np.asanyarray(img.dataobj)


def get_nifti_header_info(img):
    header = img.header
    affine = header.get_best_affine()
    dimensions = header['dim'][1:4]
    voxel_sizes = header['pixdim'][1:4]

    if not affine[0:3, 0:3].any():
        raise ValueError(
            'Invalid affine, contains only zeros.'
            'Cannot determine voxel order from transformation')
    voxel_order = ''.join(nib.aff2axcodes(affine))
    return affine, dimensions, voxel_sizes, voxel_order


def summarize_intensities(data):
    total_voxel = np.prod(data.shape)
    non_zeros = np.count_nonzero(data)
    mean = np.mean(data[data > 0])
    std = np.std(data[data > 0])
    median = np.percentile(data[data > 0], 50)
    iqr = np.percentile(data[data > 0], 75) - \
        np.percentile(data[data > 0], 25)
    max_val = np.max(data[data > 0])
    min_val = np.min(data[data > 0])
    return total_voxel, non_zeros, \
            round(mean, 3), round(std, 3), \
            round(median, 3), round(iqr, 3), \
            round(max_val, 3), round(min_val, 3)


def generate_square(shape, corner, size):
    data = np.zeros(shape)
    x_min = int(corner[0])
    y_min = int(corner[1])
    x_max = int(x_min + size[0])
    y_max = int(y_min + size[1])
    data[x_min:x_max, y_min:y_max] = 1

    return data
