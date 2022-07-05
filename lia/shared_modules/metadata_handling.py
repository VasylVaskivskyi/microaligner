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

import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Union

import tifffile as tif

from .dtype_aliases import Shape2D
from .utils import path_to_str

XML = ET.Element


class DatasetStructure:
    def __init__(self):
        self._ref_ch = "DAPI"
        self.img_paths = []
        self.input_is_stack = False
        self.output_is_stack = True

    @property
    def ref_channel_name(self) -> str:
        return self._ref_ch

    @ref_channel_name.setter
    def ref_channel_name(self, channel_name: str):
        self._ref_ch = self._strip_cycle_info(channel_name)

    def get_dataset_structure(self) -> dict:
        if self.input_is_stack:
            return self._get_stack_structure(self.img_paths[0])
        else:
            return self._get_img_list_structure(self.img_paths)

    def _str_to_xml(self, xmlstr: str) -> XML:
        """Converts str to xml and strips namespaces"""
        it = ET.iterparse(StringIO(xmlstr))
        for _, el in it:
            _, _, el.tag = el.tag.rpartition("}")
        root = it.root
        return root

    def _collect_info_from_ome(self, ome_xml: XML) -> Dict[str, Any]:
        channel_info = self._extract_channel_info(ome_xml)
        cleaned_channel_names, ref_ids = self._find_where_ref_channel(channel_info)
        pixels_info = self._extract_pixels_info(ome_xml)
        ome_info = channel_info.copy()
        ome_info["ref_ch_ids"] = ref_ids
        ome_info.update(pixels_info)
        return ome_info

    def _extract_channel_info(self, ome_xml: XML) -> Dict[str, Any]:
        channels = ome_xml.find("Image").find("Pixels").findall("Channel")
        channel_names = [ch.get("Name") for ch in channels]
        channel_fluors = []
        for ch in channels:
            if "Fluor" in ch.attrib:
                channel_fluors.append(ch.get("Fluor"))
        image_attribs = ome_xml.find("Image").find("Pixels").attrib
        nchannels = int(image_attribs.get("SizeC", 1))
        nzplanes = int(image_attribs.get("SizeZ", 1))
        channel_info = {
            "channels": channels,
            "channel_names": channel_names,
            "channel_fluors": channel_fluors,
            "nchannels": nchannels,
            "nzplanes": nzplanes,
        }
        return channel_info

    def _extract_pixels_info(self, ome_xml: XML) -> Dict[str, Union[int, float]]:
        dims = ["SizeX", "SizeY", "SizeC", "SizeZ", "SizeT"]
        sizes = ["PhysicalSizeX", "PhysicalSizeY"]
        pixels = ome_xml.find("Image").find("Pixels")
        pixels_info = dict()
        for d in dims:
            pixels_info[d] = int(pixels.get(d, 1))
        for s in sizes:
            pixels_info[s] = float(pixels.get(s, 1))
        return pixels_info

    def _strip_cycle_info(self, name) -> str:
        ch_name = re.sub(r"^(c|cyc|cycle)\d+(\s+|_)", "", name)  # strip start
        ch_name2 = re.sub(r"(-\d+)?(_\d+)?$", "", ch_name)  # strip end
        return ch_name2

    def _filter_ref_channel_ids(self, channels: List[str]) -> List[int]:
        ref_ids = []
        for _id, ch in enumerate(channels):
            if re.match(self._ref_ch, ch, re.IGNORECASE):
                ref_ids.append(_id)
        return ref_ids

    def _find_where_ref_channel(self, channel_info):
        """Find if reference channel is in fluorophores or channel names and return them"""
        channel_fluors = channel_info["channel_fluors"]
        channel_names = channel_info["channel_names"]
        # strip cycle id from channel name and fluor name
        if channel_fluors != []:
            fluors = [
                self._strip_cycle_info(fluor) for fluor in channel_fluors
            ]  # remove cycle name
        else:
            fluors = None
        names = [self._strip_cycle_info(name) for name in channel_names]

        # check if reference channel is present somewhere
        if self._ref_ch in names:
            cleaned_channel_names = names
        elif fluors is not None and self._ref_ch in fluors:
            cleaned_channel_names = fluors
        else:
            if fluors is not None:
                msg = (
                    f"Incorrect reference channel {self._ref_ch}. "
                    + f"Available channel names: {set(names)}, fluors: {set(fluors)}"
                )
                raise ValueError(msg)
            else:
                msg = (
                    f"Incorrect reference channel {self._ref_ch}. "
                    + f"Available channel names: {set(names)}"
                )
                raise ValueError(msg)
        ref_ch_ids = self._filter_ref_channel_ids(cleaned_channel_names)
        return cleaned_channel_names, ref_ch_ids

    def _get_stack_structure(self, img_path: Path):
        ome_xml = self._read_ome_meta_from_file(img_path)
        ome_info = self._collect_info_from_ome(ome_xml)
        ref_ch_ids = ome_info["ref_ch_ids"]
        nchannels = ome_info["nchannels"]
        nzplanes = ome_info["nzplanes"]

        nchannels_per_cycle = ref_ch_ids[1] - ref_ch_ids[0]
        ref_ch_position_in_cyc = ref_ch_ids[0]
        ncycles = nchannels // nchannels_per_cycle

        stack_structure = dict()
        tiff_page = 0
        for cyc in range(0, ncycles):
            img_structure = dict()
            for ch in range(0, nchannels_per_cycle):
                img_structure[ch] = dict()
                for z in range(0, nzplanes):
                    img_structure[ch][z] = tiff_page
                    tiff_page += 1
            stack_structure[cyc] = dict()
            stack_structure[cyc]["img_structure"] = img_structure
            stack_structure[cyc]["ref_channel_id"] = ref_ch_position_in_cyc
            stack_structure[cyc]["img_path"] = img_path
        return stack_structure

    def _read_ome_meta_from_file(self, path: Path) -> XML:
        with tif.TiffFile(path_to_str(path)) as TF:
            ome_meta_str = TF.ome_metadata
        ome_xml = self._str_to_xml(ome_meta_str)
        return ome_xml

    def _get_img_list_structure(self, img_paths: List[Path]):
        img_list_structure = dict()

        for cyc, path in enumerate(img_paths):
            ome_xml = self._read_ome_meta_from_file(path)
            ome_info = self._collect_info_from_ome(ome_xml)
            ref_ch_ids = ome_info["ref_ch_ids"]
            nchannels = ome_info["nchannels"]
            nzplanes = ome_info["nzplanes"]

            img_structure = dict()
            img_list_structure[cyc] = dict()
            tiff_page = 0

            for ch in range(0, nchannels):
                img_structure[ch] = dict()
                for z in range(0, nzplanes):
                    img_structure[ch][z] = tiff_page
                    tiff_page += 1

            img_list_structure[cyc]["img_structure"] = img_structure
            img_list_structure[cyc]["ref_channel_id"] = ref_ch_ids[0]
            img_list_structure[cyc]["img_path"] = path
        return img_list_structure

    def generate_new_metadata(self, target_shape: Shape2D) -> Dict[Path, str]:
        if self.output_is_stack:
            combined_meta = self.combine_meta_from_multiple_imgs(target_shape)
            return combined_meta
        else:
            updated_meta = self.update_meta_for_each_img(target_shape)
            return updated_meta

    def update_meta_for_each_img(self, target_shape: Shape2D) -> Dict[Path, str]:
        old_xmls = dict()
        planes = []
        channels = []
        for path in self.img_paths:
            old_xml = self._read_ome_meta_from_file(path)
            old_ome_info = self._collect_info_from_ome(old_xml)
            old_xmls[path] = old_xml
            planes.append(old_ome_info["SizeZ"])
            channels.append(old_ome_info["SizeC"])

        sizes = {
            "SizeX": target_shape[1],
            "SizeY": target_shape[0],
            "SizeC": channels[0],
            "SizeZ": max(planes),
            "SizeT": 1,
        }

        new_xmls = dict()
        for img_path, old_xml in old_xmls.items():
            new_xml = deepcopy(old_xml)
            for attr, val in sizes.items():
                new_xml.find("Image").find("Pixels").set(attr, str(val))

            proper_ome_attribs = {
                "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:schemaLocation": "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd",
            }
            new_xml.attrib.clear()

            for attr, val in proper_ome_attribs.items():
                new_xml.set(attr, val)

            tiffdata = new_xml.find("Image").find("Pixels").findall("TiffData")
            if tiffdata is not None or tiffdata != []:
                for td in tiffdata:
                    new_xml.find("Image").find("Pixels").remove(td)

            # add new tiffdata
            ifd = 0
            for t in range(0, sizes["SizeT"]):
                for c in range(0, sizes["SizeC"]):
                    for z in range(0, sizes["SizeZ"]):
                        ET.SubElement(
                            new_xml.find("Image").find("Pixels"),
                            "TiffData",
                            dict(
                                FirstC=str(c),
                                FirstT=str(t),
                                FirstZ=str(z),
                                IFD=str(ifd),
                                PlaneCount=str(1),
                            ),
                        )
                        ifd += 1
            xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>'
            result_ome_meta = xml_declaration + ET.tostring(
                new_xml, method="xml", encoding="utf-8"
            ).decode("ascii", errors="ignore")
            new_xmls[img_path] = result_ome_meta
        return new_xmls

    def combine_meta_from_multiple_imgs(self, target_shape: Shape2D) -> Dict[Path, str]:
        ncycles = len(self.img_paths)
        time = []
        planes = []
        channels = []
        metadata_list = []
        phys_size_x_list = []
        phys_size_y_list = []

        ome_info_list = []
        for path in self.img_paths:
            ome_xml = self._read_ome_meta_from_file(path)
            ome_info = self._collect_info_from_ome(ome_xml)
            ome_info_list.append(ome_info)
            metadata_list.append(ome_xml)
            time.append(ome_info["SizeT"])
            planes.append(ome_info["SizeZ"])
            channels.append(ome_info["SizeC"])
            phys_size_x_list.append(ome_info["PhysicalSizeX"])
            phys_size_y_list.append(ome_info["PhysicalSizeY"])

        max_time = max(time)
        max_planes = max(planes)
        total_channels = sum(channels)
        max_phys_size_x = max(phys_size_x_list)
        max_phys_size_y = max(phys_size_y_list)

        sizes = {
            "SizeX": str(target_shape[1]),
            "SizeY": str(target_shape[0]),
            "SizeC": str(total_channels),
            "SizeZ": str(max_planes),
            "SizeT": str(max_time),
            "PhysicalSizeX": str(max_phys_size_x),
            "PhysicalSizeY": str(max_phys_size_y),
        }

        # use metadata from first image as reference metadata
        ref_xml = metadata_list[0]

        # set proper ome attributes tags
        proper_ome_attribs = {
            "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd",
        }
        ref_xml.attrib.clear()

        for attr, val in proper_ome_attribs.items():
            ref_xml.set(attr, val)

        # set new dimension sizes
        for attr, size in sizes.items():
            ref_xml.find("Image").find("Pixels").set(attr, size)

        # remove old channels and tiffdata
        old_channels = ref_xml.find("Image").find("Pixels").findall("Channel")
        for ch in old_channels:
            ref_xml.find("Image").find("Pixels").remove(ch)

        tiffdata = ref_xml.find("Image").find("Pixels").findall("TiffData")
        if tiffdata is not None or tiffdata != []:
            for td in tiffdata:
                ref_xml.find("Image").find("Pixels").remove(td)

        # add new channels
        write_format = (
            "0" + str(len(str(ncycles)) + 1) + "d"
        )  # e.g. for number 5 format = 02d, result = 05
        channel_id = 0
        for i in range(0, ncycles):
            channels = ome_info_list[i]["channels"]
            channel_names = ome_info_list[i]["channel_names"]

            cycle_name = "c" + format(i + 1, write_format) + " "
            new_channel_names = [cycle_name + ch for ch in channel_names]

            for ch in range(0, len(channels)):
                new_channel_id = "Channel:0:" + str(channel_id)
                new_channel_name = new_channel_names[ch]
                channels[ch].set("Name", new_channel_name)
                channels[ch].set("ID", new_channel_id)
                ref_xml.find("Image").find("Pixels").append(channels[ch])
                channel_id += 1

        # add new tiffdata
        ifd = 0
        for t in range(0, max_time):
            for c in range(0, total_channels):
                for z in range(0, max_planes):
                    ET.SubElement(
                        ref_xml.find("Image").find("Pixels"),
                        "TiffData",
                        dict(
                            FirstC=str(c),
                            FirstT=str(t),
                            FirstZ=str(z),
                            IFD=str(ifd),
                            PlaneCount=str(1),
                        ),
                    )
                    ifd += 1

        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>'
        result_ome_meta = xml_declaration + ET.tostring(
            ref_xml, method="xml", encoding="utf-8"
        ).decode("ascii", errors="ignore")
        combined_meta = {Path("combined"): result_ome_meta}
        return combined_meta
