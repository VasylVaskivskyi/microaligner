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
from typing import List, Tuple, Union

import cv2 as cv
import numpy as np
from skimage.transform import AffineTransform, warp

from ..shared_modules.dtype_aliases import Image, TMat
from ..shared_modules.img_checks import (check_img_dims_match,
                                         check_img_is_2d_grey,
                                         check_img_is_provided)
from ..shared_modules.similarity_scoring import check_if_higher_similarity
from .feature_detection import Features
from .tile_registration import find_features, register_img_pair


class FeatureRegistrator:
    def __init__(self):
        self._ref_img = np.array([])
        self._mov_img = np.array([])
        self.num_pyr_lvl = 3
        self.num_iterations = 3
        self.tile_size = 1000
        self.use_full_res_img = False
        self.use_dog = True
        self._ref_pyr_features = []
        self._ref_img_pyr = []
        self._factors = [8, 4, 2]
        self._this_pyr_factor = 1

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

    def calc_ref_img_features(self):
        self._ref_img_pyr, self._factors = self._generate_img_pyr(self._ref_img)
        self._ref_pyr_features = []
        for pyr_level in self._ref_img_pyr:
            self._ref_pyr_features.append(
                find_features(self.dog(pyr_level, self.use_dog), self.tile_size)
            )

    def register(self, reuse_ref_img: bool = False) -> TMat:
        check_img_is_provided(self._ref_img, "ref")
        check_img_is_provided(self._mov_img, "mov")
        check_img_dims_match(self._ref_img, self._mov_img)

        if reuse_ref_img:
            if self._ref_pyr_features == []:
                self.calc_ref_img_features()
            else:
                pass
        else:
            self.calc_ref_img_features()

        mov_img_pyrs, factors = self._generate_img_pyr(self._mov_img)

        fullscale_t_mat_list = []
        for i, factor in enumerate(self._factors):
            print("Pyramid factor", factor)
            self._this_pyr_factor = factor
            if i == 0:
                mov_img_this_scale_transform, t_mat = self._iterative_alignment(
                    self._ref_img_pyr[i], self._ref_pyr_features[i], mov_img_pyrs[i]
                )
            else:
                rescaled_t_mat_list = [
                    self._rescale_t_mat(m, 1 / factor) for m in fullscale_t_mat_list
                ]
                this_scale_t_mat = self._multiply_transform_matrices(
                    rescaled_t_mat_list
                )
                mov_img_prev_scale_transform = self.transform_img(
                    mov_img_pyrs[i], this_scale_t_mat
                )
                mov_img_this_scale_transform, t_mat = self._iterative_alignment(
                    self._ref_img_pyr[i],
                    self._ref_pyr_features[i],
                    mov_img_prev_scale_transform,
                )
            fullscale_t_mat_list.append(self._rescale_t_mat(t_mat, factor))
            gc.collect()
        final_transform = self._multiply_transform_matrices(fullscale_t_mat_list)
        return final_transform

    def transform_big_img(self, img: Image, t_mat: TMat) -> Image:
        orig_dtype = deepcopy(img.dtype)
        homogenous_transform_matrix = np.append(t_mat, [[0, 0, 1]], axis=0)
        inv_matrix = np.linalg.pinv(homogenous_transform_matrix)
        AT = AffineTransform(inv_matrix)
        img = warp(img, AT, output_shape=img.shape, preserve_range=True).astype(
            orig_dtype
        )
        return img

    def transform_img(self, img: Image, t_mat: TMat) -> Image:
        if max(img.shape) > 32000:
            return self.transform_big_img(img, t_mat)
        else:
            return cv.warpAffine(img, t_mat, dsize=img.shape[::-1])

    def _generate_img_pyr(self, arr: Image) -> Tuple[List[Image], List[int]]:
        if self.num_pyr_lvl < 0:
            raise ValueError("Number of pyramid levels cannot be less than 1")
        if self.num_pyr_lvl == 0 and not self.use_full_res_img:
            msg = (
                "Number of pyramid levels is 0 and use_full_res_img is False. "
                + "Please change one of the parameters"
            )
            raise ValueError(msg)
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
        if self.use_full_res_img:
            pyramid.append(arr)
            factors.append(1)
        return pyramid, factors

    def _iterative_alignment(
        self, ref_img: Image, ref_features: Features, mov_img: Image
    ) -> Tuple[Image, TMat]:
        if self.num_iterations < 1:
            raise ValueError("Number of iterations cannot be less than 1")
        t_matrices = []
        aligned_img = mov_img.copy()
        for i in range(0, self.num_iterations):
            print("    Iteration", i + 1, "/", self.num_iterations)
            mov_img_aligned, est_t_mat_pyr = self._align_imgs(ref_features, aligned_img)

            is_more_similar = check_if_higher_similarity(
                self.dog(ref_img, True),
                self.dog(mov_img_aligned, True),
                self.dog(aligned_img, True),
                self.tile_size,
            )
            is_valid_transform = self._check_if_valid_transform(
                est_t_mat_pyr, mov_img.shape
            )

            if any(is_more_similar) and is_valid_transform:
                print("    Better alignment than before")
                t_matrices.append(est_t_mat_pyr)
                aligned_img = self._realign_img(mov_img, t_matrices)
            else:
                print("    Worse alignment than before")
                t_matrices.append(np.eye(2, 3))
                aligned_img = aligned_img
        final_t_mat = self._multiply_transform_matrices(t_matrices)
        return aligned_img, final_t_mat

    def _align_imgs(
        self, ref: Union[Image, Features], mov_img: Image
    ) -> Tuple[Image, np.ndarray]:
        if not isinstance(ref, Features):
            ref_features = find_features(self.dog(ref, self.use_dog), self.tile_size)
        else:
            ref_features = ref
        mov_features = find_features(self.dog(mov_img, self.use_dog), self.tile_size)
        transform_mat = register_img_pair(ref_features, mov_features)
        if np.equal(transform_mat, np.eye(2, 3)).all():
            return mov_img, np.eye(2, 3)
        else:
            img_aligned = self.transform_img(mov_img, transform_mat)
            return img_aligned, transform_mat

    def _realign_img(self, mov_img: Image, mat_list: List[TMat]) -> Image:
        mul_mat = self._multiply_transform_matrices(mat_list)
        img_aligned = self.transform_img(mov_img, mul_mat)
        return img_aligned

    def _multiply_transform_matrices(self, mat_list: List[TMat]) -> TMat:
        if len(mat_list) == 1:
            return mat_list[0]
        hom_mats = [np.append(mat, [[0, 0, 1]], axis=0) for mat in mat_list]
        res_mat = hom_mats[0]
        for i in range(1, len(hom_mats)):
            res_mat = res_mat @ hom_mats[i]
        res_mat_short = res_mat[:2, :]
        return res_mat_short

    def _rescale_t_mat(self, t_mat: TMat, scale: float) -> TMat:
        t_mat_copy = t_mat.copy()
        t_mat_copy[0, 2] *= scale
        t_mat_copy[1, 2] *= scale
        return t_mat_copy

    def _check_if_valid_transform(
        self, t_mat: TMat, img_shape: Tuple[int, int]
    ) -> bool:
        is_inside_border = self._check_if_inside_borders(t_mat, img_shape)
        is_proper_scale = self._check_if_proper_scale(t_mat)
        if all((is_inside_border, is_proper_scale)):
            return True
        else:
            return False

    def _check_if_proper_scale(self, t_mat: TMat) -> bool:
        # https://frederic-wang.fr/decomposition-of-2d-transform-matrices.html
        # |a c e|
        # |b d f|
        a = t_mat[0, 0]
        b = t_mat[1, 0]
        c = t_mat[0, 1]
        d = t_mat[1, 1]

        det = a * d - b * c
        if a != 0 or b != 0:
            r = np.sqrt(a**2 + b**2)
            scale = (r, det / r)
        elif c != 0 or d != 0:
            s = np.sqrt(c**2 + d**2)
            scale = (det / s, s)
        else:
            scale = (0, 0)

        if scale == (0, 0):
            return False
        elif abs(scale[0]) > 3 or abs(scale[1]) > 3:
            return False
        elif abs(scale[0]) < 0.3 or abs(scale[1]) < 0.3:
            return False
        else:
            return True

    def _check_if_inside_borders(self, t_mat: TMat, img_shape: Tuple[int, int]) -> bool:
        cy = img_shape[0] // 2
        cx = img_shape[1] // 2
        center_coords = np.array([[cx], [cy], [1]])
        border_coords = np.array([[img_shape[1]], [img_shape[0]], [1]])
        t_mat_hom = np.append(t_mat, [[0, 0, 1]], axis=0)
        transf_center = t_mat_hom @ center_coords
        if np.any((border_coords - np.abs(transf_center)) < 0):
            return False
        else:
            return True

    def get_dog_sigmas(self, pyr_factor: int) -> Tuple[int, int]:
        if pyr_factor > 16:
            return 1, 2
        else:
            sigmas = {1: (5, 9), 2: (4, 7), 4: (3, 5), 8: (2, 3), 16: (1, 2)}
        return sigmas[pyr_factor]

    def dog(self, img: Image, use_it: bool, low_sigma: int = 5, high_sigma: int = 9) -> Image:
        """Difference of Gaussian filters"""
        if not use_it:
            return img
        else:
            if img.max() == 0:
                return img
            else:
                # low_sigma, high_sigma = self.get_dog_sigmas(self._this_pyr_factor)

                fimg = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
                kernel = (low_sigma * 4 * 2 + 1, low_sigma * 4 * 2 + 1)  # as in opencv
                ls = cv.GaussianBlur(
                    fimg, kernel, sigmaX=low_sigma, dst=None, sigmaY=low_sigma
                )
                hs = cv.GaussianBlur(
                    fimg, kernel, sigmaX=high_sigma, dst=None, sigmaY=high_sigma
                )
                diff_of_gaussians = hs - ls
                del hs, ls
                diff_of_gaussians = cv.normalize(
                    diff_of_gaussians, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
                )
                return diff_of_gaussians
