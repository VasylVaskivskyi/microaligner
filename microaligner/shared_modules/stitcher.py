#  Copyright (C) 2022 Vasyl Vaskivskyi
#  MicroAligner: image registration for large scale microscopy
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

from typing import List, Tuple, Union

import numpy as np

from .dtype_aliases import Image


def get_slices(
    big_image: Image,
    hor_f: int,
    hor_t: int,
    ver_f: int,
    ver_t: int,
    padding: dict,
    overlap=0,
):
    y_axis = 0
    x_axis = 1
    # check if tile is over image boundary
    left_check = hor_f - padding["left"]
    top_check = ver_f - padding["top"]
    right_check = hor_t - big_image.shape[x_axis]
    bot_check = ver_t - big_image.shape[y_axis]

    left_pad_size = 0
    top_pad_size = 0
    right_pad_size = 0
    bot_pad_size = 0

    if left_check < 0:
        left_pad_size = abs(left_check)
        hor_f = 0
    if top_check < 0:
        top_pad_size = abs(top_check)
        ver_f = 0
    if right_check > 0:
        right_pad_size = right_check
        hor_t = big_image.shape[x_axis]
    if bot_check > 0:
        bot_pad_size = bot_check
        ver_t = big_image.shape[y_axis]

    big_image_slice = [slice(ver_f, ver_t), slice(hor_f, hor_t)]
    tile_shape = (ver_t - ver_f, hor_t - hor_f)
    tile_slice = [
        slice(top_pad_size + overlap, tile_shape[-2] + overlap),
        slice(left_pad_size + overlap, tile_shape[-1] + overlap),
    ]
    if len(big_image.shape) > 2:
        big_image_slice.append(slice(None))
        tile_slice.append(slice(None))
    return tuple(big_image_slice), tuple(tile_slice)


def stitch_image(img_list: List[Image], slicer_info: dict) -> Image:

    x_ntiles = slicer_info["ntiles"]["x"]
    y_ntiles = slicer_info["ntiles"]["y"]
    tile_shape = slicer_info["tile_shape"]
    overlap = slicer_info["overlap"]
    padding = slicer_info["padding"]

    x_axis = 1
    y_axis = 0
    ch_axis = 2

    tile_x_size = tile_shape[x_axis]
    tile_y_size = tile_shape[y_axis]

    big_image_x_size = (x_ntiles * tile_x_size) - padding["left"] - padding["right"]
    big_image_y_size = (y_ntiles * tile_y_size) - padding["top"] - padding["bottom"]

    if len(img_list[0].shape) == 2:
        big_image_shape = (big_image_y_size, big_image_x_size)
    elif len(img_list[0].shape) == 3:
        # must be a flow map
        big_image_shape = (big_image_y_size, big_image_x_size, 2)
    else:
        msg = f"Input image has unexpected dimensions - {img_list[0].shape}"
        raise ValueError(msg)
    dtype = img_list[0].dtype
    big_image = np.zeros(big_image_shape, dtype=dtype)

    n = 0
    for i in range(0, y_ntiles):
        ver_f = i * tile_y_size
        ver_t = ver_f + tile_y_size

        for j in range(0, x_ntiles):
            hor_f = j * tile_x_size
            hor_t = hor_f + tile_x_size

            big_image_slice, tile_slice = get_slices(
                big_image, hor_f, hor_t, ver_f, ver_t, padding, overlap
            )
            tile = img_list[n]

            big_image[tuple(big_image_slice)] = tile[tuple(tile_slice)]
            n += 1

    return big_image
