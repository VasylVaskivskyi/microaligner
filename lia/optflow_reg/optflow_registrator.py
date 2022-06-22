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
from typing import List, Tuple

import cv2 as cv
import dask
import numpy as np

from ..shared_modules.dtype_aliases import Flow, Image
from ..shared_modules.img_checks import (check_img_is_2d_grey,
                                         check_img_is_provided)
from ..shared_modules.similarity_scoring import check_if_higher_similarity
from ..shared_modules.slicer import split_image_into_tiles_of_size
from ..shared_modules.stitcher import stitch_image
from .flow_calc import TileFlowCalc
from .warper import Warper


def merge_two_flows(flow1: Flow, flow2: Flow) -> Flow:
    # https://openresearchsoftware.metajnl.com/articles/10.5334/jors.380/
    # m_flow = of.combine_flows(flow1, flow2, 3, ref="s")
    if flow1.max() == 0:
        return flow2
    elif flow2.max() == 0:
        return flow1
    else:
        m_flow = flow1 + cv.remap(flow2, -flow1, None, cv.INTER_LINEAR)
    gc.collect()
    return m_flow


class OptFlowRegistrator:
    def __init__(self):
        self._ref_img = np.array([])
        self._mov_img = np.array([])
        self.num_pyr_lvl = 4
        self.num_iterations = 3
        self.tile_size = 1000
        self.overlap = 100
        self._warper = Warper()
        self._tile_flow_calc = TileFlowCalc()

    @property
    def ref_img(self) -> Image:
        return self._ref_img

    @ref_img.setter
    def ref_img(self, img: Image):
        check_img_is_2d_grey(img, "ref")
        self._ref_img = img

    @property
    def mov_img(self) -> Image:
        return self._ref_img

    @mov_img.setter
    def mov_img(self, img: Image):
        check_img_is_2d_grey(img, "mov")
        self._mov_img = img

    def _init_warper(self):
        self._warper = Warper()
        self._warper.tile_size = self.tile_size
        self._warper.overlap = self.overlap

    def _init_tile_flow_calc(self):
        self._tile_flow_calc = TileFlowCalc()
        self._tile_flow_calc.tile_size = self.tile_size
        self._tile_flow_calc.overlap = self.overlap
        self._tile_flow_calc.num_iter = self.num_iterations
        self._tile_flow_calc.win_size = self.overlap - (1 - self.overlap % 2)

    def register(self) -> np.ndarray:
        check_img_is_provided(self._ref_img, "ref")
        check_img_is_provided(self._mov_img, "mov")

        self._init_tile_flow_calc()
        self._init_warper()

        ref_pyr, factors = self._generate_img_pyr(self._ref_img)
        mov_pyr, f_ = self._generate_img_pyr(self._mov_img)

        for lvl, factor in enumerate(factors):
            print("Pyramid factor", factor)
            mov_this_lvl = mov_pyr[lvl].copy()

            # apply previous flow
            if lvl == 0:
                pass
            else:
                self._warper.image = mov_this_lvl
                self._warper.flow = m_flow
                mov_this_lvl = self._warper.warp()
                # mov_this_lvl = self.warp_with_flow(mov_this_lvl, m_flow)
            self._tile_flow_calc.ref_img = ref_pyr[lvl]
            self._tile_flow_calc.mov_img = mov_this_lvl
            this_flow = self._tile_flow_calc.calc_flow()
            self._warper.image = mov_this_lvl
            self._warper.flow = this_flow
            mov_this_lvl = self._warper.warp()
            gc.collect()
            # this_flow = self.calc_flow(ref_pyr[lvl], mov_this_lvl, 1, 0, 51)
            is_higher_similarity = check_if_higher_similarity(
                ref_pyr[lvl], mov_this_lvl, mov_pyr[lvl], self.tile_size
            )
            if not any(is_higher_similarity):
                print("    Worse alignment than before")
                if lvl == 0:
                    dstsize = list(mov_pyr[lvl + 1].shape)
                    m_flow = np.zeros(dstsize + [2], dtype=np.float32)
                else:
                    dstsize = mov_pyr[lvl + 1].shape[::-1]
                    m_flow = cv.pyrUp(m_flow * 2, dstsize=dstsize)
            else:
                print("    Better alignment than before")
                # merge flows, upscale to next level
                if lvl == 0:
                    if len(factors) > 1:
                        dstsize = mov_pyr[lvl + 1].shape[::-1]
                        m_flow = cv.pyrUp(this_flow * 2, dstsize=dstsize)
                    else:
                        m_flow = this_flow
                elif lvl == len(factors) - 1:
                    m_flow = self._merge_list_of_flows([m_flow, this_flow])
                else:
                    m_flow = self._merge_list_of_flows([m_flow, this_flow])
                    dstsize = mov_pyr[lvl + 1].shape[::-1]
                    m_flow = cv.pyrUp(m_flow * 2, dstsize=dstsize)
                del this_flow
        del mov_pyr, ref_pyr
        gc.collect()
        return m_flow

    def _generate_img_pyr(self, arr: Image) -> Tuple[List[Image], List[int]]:
        # Pyramid scales from smallest to largest
        if self.num_pyr_lvl < 0:
            raise ValueError("Number of pyramid levels cannot be less than 0")
        # Pyramid scales from smallest to largest
        pyramid: List[Image] = []
        factors = []
        pyr_lvl = arr.copy()
        for lvl in range(0, self.num_pyr_lvl):
            factor = 2 ** (lvl + 1)
            if arr.shape[0] / factor < 100 or arr.shape[1] / factor < 100:
                break
            else:
                pyramid.append(cv.pyrDown(pyr_lvl))
                pyr_lvl = pyramid[lvl]
                factors.append(factor)
        factors = list(reversed(factors))
        pyramid = list(reversed(pyramid))
        pyramid.append(arr)
        factors.append(1)
        return pyramid, factors

    def _merge_flow_in_tiles(self, flow1: Flow, flow2: Flow):
        flow1_list, slicer_info = split_image_into_tiles_of_size(
            flow1, self.tile_size, self.tile_size, self.overlap
        )
        flow2_list, s_ = split_image_into_tiles_of_size(
            flow2, self.tile_size, self.tile_size, self.overlap
        )
        del flow1, flow2

        tasks = []
        for i in range(0, len(flow1_list)):
            task = dask.delayed(merge_two_flows)(flow1_list[i], flow2_list[i])
            tasks.append(task)
        merged_flow_tiles = dask.compute(*tasks)
        del flow1_list, flow2_list
        merged_flow = stitch_image(merged_flow_tiles, slicer_info)
        return merged_flow

    def _merge_list_of_flows(self, flow_list: List[Flow]) -> Flow:
        m_flow = flow_list[0]
        if len(flow_list) > 1:
            for i in range(1, len(flow_list)):
                m_flow = self._merge_flow_in_tiles(m_flow, flow_list[i])
        return m_flow