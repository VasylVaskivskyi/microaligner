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

from typing import List, Tuple

import dask
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from .dtype_aliases import Image


def mi_tiled(arr1: Image, arr2: Image, tile_size: int) -> float:
    """Calculates mutual information score for two input arrays.
    To save time and memory it splits image into tiles and
    does calculation in parallel.
    It must remain separate from the registrator classes,
    otherwise it will bloat memory, because dask copies
    contents of the whole class into the memory.
    """
    if max(arr1.shape) / tile_size < 2:
        return normalized_mutual_info_score(arr1.flatten(), arr2.flatten())
    else:
        indices = list(range(0, arr1.size, tile_size * tile_size))
        arr1_parts = np.array_split(arr1.flatten(), indices)
        arr2_parts = np.array_split(arr2.flatten(), indices)
        tasks = []
        for i in range(0, len(arr1_parts)):
            if arr1_parts[i].size != 0:
                task = dask.delayed(normalized_mutual_info_score)(
                    arr1_parts[i], arr2_parts[i]
                )
                tasks.append(task)
        scores = dask.compute(*tasks)
        mi_score = np.mean(scores)
    return mi_score


def mutual_information_test(
    ref_arr: Image, test_arr: Image, init_arr: Image, tile_size: int
) -> Tuple[float, float]:
    after_mi_score = mi_tiled(ref_arr, test_arr, tile_size)
    before_mi_score = mi_tiled(ref_arr, init_arr, tile_size)
    return after_mi_score, before_mi_score


def check_if_higher_similarity(
    ref_arr: Image, test_arr: Image, init_arr: Image, tile_size: int
) -> List[bool]:
    mi_scores = mutual_information_test(ref_arr, test_arr, init_arr, tile_size)
    checks = list()
    checks.append(mi_scores[0] > mi_scores[1])
    print("    MI score after:", mi_scores[0], "| MI score before:", mi_scores[1])
    return checks
