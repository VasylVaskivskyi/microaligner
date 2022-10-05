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

import gc
from typing import List, Tuple

import cv2 as cv
import numpy as np

from ..shared_modules.dtype_aliases import Flow, Image
from ..shared_modules.slicer import split_image_into_tiles_of_size
from ..shared_modules.stitcher import stitch_image


class Warper:
    def __init__(self):
        self.image = np.array([])
        self.flow = np.array([])
        self.tile_size = 1000
        self.overlap = 100
        self._slicer_info = {}

    def warp(self):
        image_tiles, self._slicer_info = split_image_into_tiles_of_size(
            self.image, self.tile_size, self.tile_size, self.overlap
        )
        self.image = np.array([])
        flow_tiles, s_ = split_image_into_tiles_of_size(
            self.flow, self.tile_size, self.tile_size, self.overlap
        )
        self.flow = np.array([])
        warped_image_tiles = self._warp_image_tiles(image_tiles, flow_tiles)
        del image_tiles, flow_tiles
        stitched_warped_image = stitch_image(warped_image_tiles, self._slicer_info)

        self._slicer_info = {}
        del warped_image_tiles
        gc.collect()
        return stitched_warped_image

    def _make_flow_for_remap(self, flow: Flow) -> Flow:
        h, w = flow.shape[:2]
        new_flow = np.negative(flow)
        new_flow[:, :, 0] += np.arange(w)
        new_flow[:, :, 1] += np.arange(h).reshape(-1, 1)
        return new_flow

    def _warp_with_flow(self, img: Image, flow: Flow) -> Image:
        """Warps input image according to optical flow"""
        new_flow = self._make_flow_for_remap(flow)
        res = cv.remap(img, new_flow, None, cv.INTER_LINEAR)
        gc.collect()
        return res

    def _warp_image_tiles(
        self, image_tiles: List[Image], flow_tiles: List[np.ndarray]
    ) -> List[Image]:
        warped_tiles = []
        # parallelizing this loop is not worth it - it only increases memory consumption and processing time
        for t in range(0, len(image_tiles)):
            warped_tiles.append(self._warp_with_flow(image_tiles[t], flow_tiles[t]))
        return warped_tiles
