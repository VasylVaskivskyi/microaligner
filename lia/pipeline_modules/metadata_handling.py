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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import tifffile as tif

from ..shared_modules.dtype_aliases import XML, Shape2D
from .ome_meta_processing import (_strip_cycle_info, collect_info_from_ome,
                                  read_ome_meta_from_file)
from .stack_builder import generate_ome_for_cycle_builder
from ..shared_modules.utils import path_to_str


@dataclass
class DatasetStruct:
    tiff_pages: Dict[int, Dict[int, Dict[int, int]]] = field(default_factory=dict)
    img_paths: Dict[int, Dict[int, Dict[int, Path]]] = field(default_factory=dict)
    ref_channel_ids: Dict[int, int] = field(default_factory=dict)
    ome_xmls: Dict[int, XML] = field(default_factory=dict)


class DatasetStructCreator:
    def __init__(self):
        self._ref_ch = "DAPI"
        self.img_paths: Union[None, Path, Dict[int, Path], Dict[int, Dict[str, Path]]] = None
        self.input_is_stack = False
        self.input_is_stack_builder = False
        self.output_is_stack = True

    @property
    def ref_channel_name(self) -> str:
        return self._ref_ch

    @ref_channel_name.setter
    def ref_channel_name(self, channel_name: str):
        self._ref_ch = _strip_cycle_info(channel_name)

    def create_dataset_struct(self) -> DatasetStruct:
        if self.img_paths is None:
            raise ValueError("Attribute img_paths is empty")

        if self.input_is_stack:
            return self._get_stack_structure(self.img_paths)
        elif self.input_is_stack_builder:
            return self._get_stack_builder_structure(self.img_paths)
        else:
            return self._get_img_list_structure(self.img_paths)

    def _get_stack_builder_structure(
        self, cycle_map: Dict[int, Dict[str, Path]]
    ) -> DatasetStruct:
        ome_meta_dict = generate_ome_for_cycle_builder(cycle_map)
        stack_builder_structure = DatasetStruct()
        for cyc, ome_xml in ome_meta_dict.items():
            img_path_per_ch = cycle_map[cyc]
            ome_info = collect_info_from_ome(self._ref_ch, ome_xml)
            ref_ch_ids = ome_info["ref_ch_ids"]
            nchannels = ome_info["nchannels"]
            nzplanes = ome_info["nzplanes"]

            tiff_pages: Dict[int, Dict[int, int]] = dict()
            img_paths: Dict[int, Dict[int, Path]] = dict()

            ch_names = list(img_path_per_ch.keys())
            for ch in range(1, nchannels + 1):
                tiff_pages[ch] = dict()
                img_paths[ch] = dict()
                ch_name = ch_names[ch - 1]
                img_path = img_path_per_ch[ch_name]
                for z in range(1, nzplanes + 1):
                    tiff_pages[ch][z] = z
                    img_paths[ch][z] = img_path

            stack_builder_structure.tiff_pages[cyc] = tiff_pages
            stack_builder_structure.img_paths[cyc] = img_paths
            stack_builder_structure.ref_channel_ids[cyc] = ref_ch_ids[0] + 1
            stack_builder_structure.ome_xmls[cyc] = ome_xml
        return stack_builder_structure

    def _get_stack_structure(self, cycle_stack_path: Dict[int, Path]) -> DatasetStruct:
        first_key = sorted(list(cycle_stack_path.keys()))[0]
        img_path = cycle_stack_path[first_key]
        ome_xml = read_ome_meta_from_file(img_path)
        ome_info = collect_info_from_ome(self._ref_ch, ome_xml)
        ref_ch_ids = ome_info["ref_ch_ids"]
        nchannels = ome_info["nchannels"]
        nzplanes = ome_info["nzplanes"]

        nchannels_per_cycle = ref_ch_ids[1] - ref_ch_ids[0]
        ref_ch_position_in_cyc = ref_ch_ids[0]
        ncycles = nchannels // nchannels_per_cycle

        stack_structure = DatasetStruct()
        tiff_page = 0
        for cyc in range(1, ncycles + 1):
            tiff_pages: Dict[int, Dict[int, int]] = dict()
            img_paths: Dict[int, Dict[int, Path]] = dict()
            for ch in range(1, nchannels_per_cycle + 1):
                tiff_pages[ch] = dict()
                img_paths[ch] = dict()
                for z in range(1, nzplanes + 1):
                    tiff_pages[ch][z] = tiff_page
                    img_paths[ch][z] = img_path
                    tiff_page += 1

            stack_structure.tiff_pages[cyc] = tiff_pages
            stack_structure.img_paths[cyc] = img_paths
            stack_structure.ref_channel_ids[cyc] = ref_ch_position_in_cyc + 1
            stack_structure.ome_xmls[cyc] = ome_xml
        return stack_structure

    def _get_img_list_structure(self, img_paths: Dict[int, Path]) -> DatasetStruct:
        img_list_structure = DatasetStruct()

        for cyc, img_path in img_paths.items():
            ome_xml = read_ome_meta_from_file(img_path)
            ome_info = collect_info_from_ome(self._ref_ch, ome_xml)
            ref_ch_ids = ome_info["ref_ch_ids"]
            nchannels = ome_info["nchannels"]
            nzplanes = ome_info["nzplanes"]

            tiff_pages: Dict[int, Dict[int, int]] = dict()
            img_paths: Dict[int, Dict[int, Path]] = dict()

            tiff_page = 0
            for ch in range(1, nchannels + 1):
                tiff_pages[ch] = dict()
                img_paths[ch] = dict()
                for z in range(1, nzplanes + 1):
                    tiff_pages[ch][z] = tiff_page
                    img_paths[ch][z] = img_path
                    tiff_page += 1

            img_list_structure.tiff_pages[cyc] = tiff_pages
            img_list_structure.img_paths[cyc] = img_paths
            img_list_structure.ref_channel_ids[cyc] = ref_ch_ids[0] + 1
            img_list_structure.ome_xmls[cyc] = ome_xml
        return img_list_structure
