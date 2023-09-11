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
from scipy.stats import rankdata
import cv2 as cv

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

def pearson_cor_coef(arr1, arr2) -> float:
    X = arr1.flatten()
    Y = arr2.flatten()
    X_bar = np.average(X)
    Y_bar = np.average(Y)
    pearson = np.sum((X-X_bar)*(Y-Y_bar)) / (np.sqrt(np.sum((X-X_bar)**2))*(np.sqrt(np.sum((Y-Y_bar)**2))) + 1e-6)
    return pearson


def pcc_test(ref_arr: Image, test_arr: Image, init_arr: Image) -> Tuple[float, float]:
    after_pcc_score = pearson_cor_coef(ref_arr, test_arr)
    before_pcc_score = pearson_cor_coef(ref_arr, init_arr)
    return after_pcc_score, before_pcc_score

def spearman_cor_coef(arr1, arr2) -> float:
    X = arr1.flatten()
    Y = arr2.flatten()
    Xr = rankdata(X).astype(np.uint32)
    Yr = rankdata(Y).astype(np.uint32)
    Xr_bar = np.average(Xr)
    Yr_bar = np.average(Yr)
    spearman = np.sum((Xr - Xr_bar) * (Yr - Yr_bar)) / (
        np.sqrt(np.sum((Xr - Xr_bar) ** 2)) * (np.sqrt(np.sum((Yr - Yr_bar) ** 2))) + 1e-6
    )
    return spearman

def scc_test(ref_arr: Image, test_arr: Image, init_arr: Image) -> Tuple[float, float]:
    after_scc_score = spearman_cor_coef(ref_arr, test_arr)
    before_scc_score = spearman_cor_coef(ref_arr, init_arr)
    return after_scc_score, before_scc_score


def mutual_information_test(
    ref_arr: Image, test_arr: Image, init_arr: Image, tile_size: int
) -> Tuple[float, float]:
    after_mi_score = mi_tiled(ref_arr, test_arr, tile_size)
    before_mi_score = mi_tiled(ref_arr, init_arr, tile_size)
    return after_mi_score, before_mi_score

def ecc_test(ref_arr: Image, test_arr: Image, init_arr: Image) -> Tuple[float, float]:
    after_ecc_score = cv.computeECC(ref_arr, test_arr)
    before_ecc_score = cv.computeECC(ref_arr, init_arr)
    return after_ecc_score, before_ecc_score

def check_if_higher_similarity(
    ref_arr: Image, test_arr: Image, init_arr: Image, tile_size: int
) -> bool:
    mi_scores = mutual_information_test(ref_arr, test_arr, init_arr, tile_size)
    # scc_scores = scc_test(ref_arr, test_arr, init_arr)
    # ecc_scores = ecc_test(ref_arr, test_arr, init_arr)

    mi_check = mi_scores[0] > mi_scores[1]
    # scc_check = scc_scores[0] > scc_scores[1]
    # ecc_check = ecc_scores[0] > ecc_scores[1]

    check_result = mi_check # sum([scc_check, mi_check, ecc_check]) > 1

    mi_mark = "[+]" if mi_check else "[-]"
    # scc_mark = "[+]" if scc_check else "[-]"
    # ecc_mark = "[+]" if ecc_check else "[-]"

    print("    ", mi_mark, "MI score after:", mi_scores[0], "| MI score before:", mi_scores[1])
    # print("    ", scc_mark, "SCC score after:", scc_scores[0], "| SCC score before:", scc_scores[1])
    # print("    ", ecc_mark, "ECC score after:", ecc_scores[0], "| ECC score before:", ecc_scores[1])
    return check_result

