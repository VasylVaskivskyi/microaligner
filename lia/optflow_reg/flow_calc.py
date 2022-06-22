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
from typing import List

import cv2 as cv
import dask
import numpy as np

from ..shared_modules.dtype_aliases import Flow, Image
from ..shared_modules.slicer import split_image_into_tiles_of_size
from ..shared_modules.stitcher import stitch_image


def farneback(
    mov_img: Image, ref_img: Image, pyr_size=0, win_size=51, num_iter=1
) -> Flow:
    flow = cv.calcOpticalFlowFarneback(
        mov_img,
        ref_img,
        None,
        pyr_scale=0.5,
        levels=pyr_size,
        winsize=win_size,
        iterations=num_iter,
        poly_n=1,
        poly_sigma=1.7,
        flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    # large values of poly_n produce smudges
    gc.collect()
    return flow


class TileFlowCalc:
    def __init__(self):
        self.ref_img = np.array([])
        self.mov_img = np.array([])
        self.num_iter = 1
        self.win_size = 51
        self.tile_size = 1000
        self.overlap = 100

    def calc_flow(self) -> Flow:
        max_dim = max(self.ref_img.shape)
        if max_dim / self.tile_size < 2:
            stitched_flow = self._calc_flow_one_pair(
                self.mov_img, self.ref_img, self.win_size, self.num_iter
            )
        else:
            ref_img_tiles, slicer_info = split_image_into_tiles_of_size(
                self.ref_img, self.tile_size, self.tile_size, self.overlap
            )
            self.ref_img = np.array([])
            mov_img_tiles, s_ = split_image_into_tiles_of_size(
                self.mov_img, self.tile_size, self.tile_size, self.overlap
            )
            self.mov_img = np.array([])
            flow_tiles = self._calc_flow_for_tile_pairs(ref_img_tiles, mov_img_tiles)
            del ref_img_tiles, mov_img_tiles
            stitched_flow = stitch_image(flow_tiles, slicer_info)
            del flow_tiles
        gc.collect()
        return stitched_flow

    def _calc_flow_one_pair(
        self, mov_img: Image, ref_img: Image, win_size: int, num_iter: int
    ) -> Flow:
        flow = farneback(mov_img, ref_img, 0, win_size, num_iter)
        gc.collect()
        return flow

    def _calc_flow_for_tile_pairs(
        self, ref_img_tiles: List[Image], mov_img_tiles: List[Image]
    ) -> List[Flow]:
        tasks = []
        for i in range(0, len(ref_img_tiles)):
            task = dask.delayed(farneback)(
                mov_img_tiles[i], ref_img_tiles[i], 0, self.win_size, self.num_iter
            )
            tasks.append(task)
        flow_tiles = dask.compute(*tasks)
        return flow_tiles
