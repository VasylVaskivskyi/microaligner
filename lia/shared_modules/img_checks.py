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

from pathlib import Path
from typing import List

import tifffile as tif

from .dtype_aliases import Image


def check_img_is_2d_grey(img: Image, img_type: str):
    if len(img.shape) > 2:
        msg = (
            f"Expected {str(img_type)} input to be 2D grayscale image, "
            + f"but received {str(img_type)} image with shape {img.shape}"
        )
        raise ValueError(msg)


def check_img_is_provided(img: Image, img_type: str):
    if len(img) == 0:
        msg = f"No {str(img_type)} image provided"
        raise ValueError(msg)


def check_img_dims_match(ref: Image, mov: Image):
    if ref.shape != mov.shape:
        msg = (
            "Input image have different dimensions"
            + f"reference image shape: {ref.shape}, moving image shape: {mov.shape}"
        )
        raise ValueError(msg)


def check_input_has_proper_dimensions(img_path: Path):
    """Image has to have dimension order CZYX,
    and doesn't have any extra dimensions (T,S, etc)
    """
    TF = tif.TiffFile(img_path)
    img_shape = TF.series[0].shape
    num_dims = len(img_shape)
    if num_dims != 4:
        msg = (
            "Expected image that has precisely 4 dimensions "
            + f"but image {str(img_path)} has {num_dims}, "
            + f"and shape {img_shape}"
        )
        raise ValueError(msg)
    else:
        pass


def check_number_of_input_img_paths(img_paths: List[Path], is_stack: bool):
    if len(img_paths) == 1:
        if is_stack:
            pass
        else:
            msg = "You need to provide at least two images to do a registration."
            raise ValueError(msg)
    elif len(img_paths) > 1:
        if is_stack:
            msg = (
                "Too many input images. "
                + "When flag InputIsCycleStack is true only one image can be used"
            )
            raise ValueError(msg)
        else:
            pass
    else:
        msg = "You need to provide at least two images to do a registration."
        raise ValueError(msg)
