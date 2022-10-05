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

import cv2 as cv
import dask
import numpy as np

from ..shared_modules.dtype_aliases import Descriptors, Image


class Features:
    def __init__(self):
        self._keypoints: cv.KeyPoint = None
        self._descriptors: Descriptors = None

    def is_valid(self) -> bool:
        if self._keypoints is None or self._descriptors is None:
            return False
        else:
            return True

    @property
    def keypoints(self) -> Tuple[Union[cv.KeyPoint, None]]:
        if self._keypoints is None:
            return None
        cv_keypoints = []
        for kp in self._keypoints:
            cv_kp = cv.KeyPoint(
                x=kp[0][0],
                y=kp[0][1],
                size=kp[1],
                angle=kp[2],
                response=kp[3],
                octave=kp[4],
                class_id=kp[5],
            )
            cv_keypoints.append(cv_kp)
        return tuple(cv_keypoints)

    @keypoints.setter
    def keypoints(self, kps: Tuple[Union[None, cv.KeyPoint]]):
        if kps is None:
            self._keypoints = None
        else:
            temp_kp_storage = []
            for point in kps:
                temp_kp_storage.append(
                    (
                        point.pt,
                        point.size,
                        point.angle,
                        point.response,
                        point.octave,
                        point.class_id,
                    )
                )
            self._keypoints = temp_kp_storage

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, des: Union[None, Descriptors]):
        self._descriptors = des


def view_tile_without_overlap(img, overlap):
    return img[overlap:-overlap, overlap:-overlap]


def find_features(img: Image, nfeatures_limit: int = 5000) -> Features:
    if img.max() == 0:
        return Features()
    # default values except for threshold - discard points that have 0 response
    detector = cv.FastFeatureDetector_create(
        threshold=1, nonmaxSuppression=True, type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16
    )
    # default values
    descriptor = cv.xfeatures2d.DAISY_create(
        radius=21,
        q_radius=3,
        q_theta=8,
        q_hist=8,
        norm=cv.xfeatures2d.DAISY_NRM_NONE,
        interpolation=True,
        use_orientation=False,
    )
    overlap = 51
    kp = detector.detect(view_tile_without_overlap(img, overlap))
    kp = sorted(kp, key=lambda x: x.response, reverse=True)[:nfeatures_limit]
    kp, des = descriptor.compute(img, kp)

    if kp is None or len(kp) < 3:
        kp = None
    if des is None or len(des) < 3:
        des = None

    features = Features()
    features.keypoints = kp
    features.descriptors = des
    return features


def match_features(img1_features: Features, img2_features: Features) -> np.ndarray:
    if not img1_features.is_valid() or not img2_features.is_valid():
        return np.eye(2, 3)
    else:
        kp1 = img1_features.keypoints
        des1 = img1_features.descriptors
        kp2 = img2_features.keypoints
        des2 = img2_features.descriptors

    FLANN_INDEX_KDTREE = 1
    index_param = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
    search_param = dict(checks=50, sorted=True, explore_all_trees=False)
    # matcher = cv.FlannBasedMatcher(index_param, search_param)
    matcher = cv.FlannBasedMatcher_create()
    matches = matcher.knnMatch(des2, des1, k=2)

    # Filter out unreliable points
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    print("    Good matches", len(good), "/", len(matches))
    if len(good) < 3:
        return np.eye(2, 3)
    # convert keypoints to format acceptable for estimator
    src_pts = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float32).reshape(
        (-1, 1, 2)
    )
    dst_pts = np.array([kp2[m.queryIdx].pt for m in good], dtype=np.float32).reshape(
        (-1, 1, 2)
    )

    # find out how images shifted (compute affine transformation)
    affine_transform_matrix, mask = cv.estimateAffinePartial2D(
        dst_pts, src_pts, method=cv.RANSAC, confidence=0.99
    )
    return affine_transform_matrix


def find_features_parallelized(tile_list: List[Image]) -> List[Features]:
    n_tiles = len(tile_list)
    nfeatures_limit_per_tile = min(1000000 // n_tiles, 5000)
    task = []
    for tile in tile_list:
        task.append(dask.delayed(find_features)(tile, nfeatures_limit_per_tile))
    tiles_features = dask.compute(*task)
    return tiles_features
