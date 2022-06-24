#  Copyright (C) 2022 Vasyl Vaskivskyi
#  LIA: Large image aligner for microscopy images
#  Email: vaskivskyi.v@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import gc
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import dask
import numpy as np
import tifffile as tif
from skimage.transform import AffineTransform, warp

from .dtype_aliases import Image, Shape2D, TMat


def path_to_str(path: Path) -> str:
    if isinstance(path, str):
        return path
    else:
        return str(path.absolute().as_posix())


def _calculate_padding_size(bigger_shape: int, smaller_shape: int) -> Tuple[int, int]:
    """Find difference between shapes of bigger and smaller image"""
    diff = bigger_shape - smaller_shape
    if diff == 1:
        dim1 = 0
        dim2 = 1
    elif diff % 2 != 0:
        dim1 = int(diff // 2)
        dim2 = int((diff // 2) + 1)
    else:
        dim1 = dim2 = int(diff / 2)
    return dim1, dim2


def pad_to_shape(
    img: Image, target_shape: Tuple[int, int]
) -> Tuple[Image, Tuple[int, int, int, int]]:
    """Will pad image to the target shape"""
    if img.shape == target_shape:
        return img, (0, 0, 0, 0)
    else:
        left, right = _calculate_padding_size(target_shape[1], img.shape[1])
        top, bottom = _calculate_padding_size(target_shape[0], img.shape[0])
        padding = (left, right, top, bottom)
        padded_img = cv.copyMakeBorder(
            img, top, bottom, left, right, cv.BORDER_CONSTANT, None, 0
        )
        return padded_img, padding


def read_and_max_project_pages(img_path: Path, tiff_pages: List[int]) -> Image:
    max_proj = tif.imread(path_to_str(img_path), key=tiff_pages[0])

    if len(tiff_pages) > 1:
        del tiff_pages[0]
        for p in tiff_pages:
            max_proj = np.maximum(max_proj, tif.imread(path_to_str(img_path), key=p))
    max_proj = cv.normalize(max_proj, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    return max_proj


def transform_img_with_tmat(
    img: Image, target_shape: Shape2D, transform_matrix: TMat
) -> Image:
    original_dtype = deepcopy(img.dtype)
    img, _ = pad_to_shape(img, target_shape)
    gc.collect()
    identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    if not np.array_equal(transform_matrix, identity_matrix):
        transform_matrix_3x3 = np.append(transform_matrix, [[0, 0, 1]], axis=0)
        # Using partial inverse to handle singular matrices
        inv_matrix = np.linalg.pinv(transform_matrix_3x3)
        AT = AffineTransform(inv_matrix)
        img = warp(img, AT, output_shape=img.shape, preserve_range=True).astype(
            original_dtype
        )
        gc.collect()
    return img


def set_number_of_dask_workers(n_workers: int = 0):
    if n_workers == 0:
        dask.config.set({"scheduler": "processes"})
    elif n_workers == 1:
        dask.config.set({"scheduler": "synchronous"})
    else:
        dask.config.set({"num_workers": n_workers, "scheduler": "processes"})
