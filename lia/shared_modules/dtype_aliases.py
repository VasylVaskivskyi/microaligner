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

import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np

# image data 2d array
Image = np.ndarray

# transformation matrix 2x3
TMat = np.ndarray

# array of image descrioptors,
# it has shape n_keypoints x n_features
Descriptors = np.ndarray

# optical flow map, 3d array n x m x 2
Flow = np.ndarray

# array shape
Shape2D = Tuple[int, int]

# values used for padding (left, right, top, bottom)
Padding = Tuple[int, int, int, int]

XML = ET.Element
