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
from io import StringIO
from typing import Dict, List
from pathlib import Path

import tifffile as tif

from ..shared_modules.dtype_aliases import XML


def str_to_xml(xmlstr: str) -> XML:
    """Converts str to xml and strips namespaces"""
    it = ET.iterparse(StringIO(xmlstr))
    for _, el in it:
        _, _, el.tag = el.tag.rpartition("}")
    root = it.root
    return root


def get_first_element_of_dict(dictionary: dict):
    dict_keys = list(dictionary.keys())
    first_key = dict_keys[0]
    return dictionary[first_key]


def digits_from_str(string: str) -> List[int]:
    return [int(x) for x in re.split(r"(\d+)", string) if x.isdigit()]


def process_cycle_map(
    cycle_map: Dict[str, Dict[str, str]]
) -> Dict[int, Dict[str, str]]:
    cycle_names = list(cycle_map.keys())
    cycle_ids = [digits_from_str(name)[0] for name in cycle_names]
    new_cycle_map = dict()
    for i in range(0, len(cycle_ids)):
        this_cycle_name = cycle_names[i]
        this_cycle_id = cycle_ids[i]
        new_cycle_map[this_cycle_id] = cycle_map[this_cycle_name]

    # sort keys
    sorted_keys = sorted(new_cycle_map.keys())
    processed_cycle_map = dict()
    for k in sorted_keys:
        processed_cycle_map[k] = new_cycle_map[k]
    return processed_cycle_map


def get_image_dims(path: Path) -> Dict[str, int]:
    with tif.TiffFile(path) as TF:
        image_shape = list(TF.series[0].shape)
        image_dims = list(TF.series[0].axes)
    dims = ["Q", "C", "Z", "Y", "X"]
    image_dimensions = dict()
    for d in dims:
        if d in image_dims:
            idx = image_dims.index(d)
            image_dimensions[d] = image_shape[idx]
        else:
            image_dimensions[d] = 1
    q_size = image_dimensions["Q"]
    c_size = image_dimensions["C"]
    z_size = image_dimensions["Z"]

    if sum((q_size > 1, c_size > 1, z_size > 1)) >= 2:
            msg = f"The input image has too many dimensions"
            raise ValueError(msg)
    else:
        higher_dims = ["Q", "C", "Z"]
        dim_val = 1
        for dim in higher_dims:
            if image_dimensions[dim] > 1:
                dim_val = image_dimensions[dim]
        for dim in higher_dims:
            del image_dimensions[dim]
        image_dimensions["Z"] = dim_val
    return image_dimensions


def get_dimensions_per_cycle(
    cycle_map: Dict[int, Dict[str, Path]]
) -> Dict[int, Dict[str, int]]:
    dimensions_per_cycle = dict()
    for cycle in cycle_map:
        this_cycle_channels = cycle_map[cycle]
        this_cycle_channels_paths = list(this_cycle_channels.values())
        num_channels = len(this_cycle_channels_paths)
        first_channel_dims = get_image_dims(this_cycle_channels_paths[0])
        num_z_planes = (
            1
            if first_channel_dims["Z"] == 1
            else first_channel_dims["Z"] * num_channels
        )
        this_cycle_dims = {
            "SizeT": 1,
            "SizeZ": num_z_planes,
            "SizeC": num_channels,
            "SizeY": first_channel_dims["Y"],
            "SizeX": first_channel_dims["X"],
        }
        dimensions_per_cycle[cycle] = this_cycle_dims
    return dimensions_per_cycle


def generate_channel_meta(channel_names: List[str], cycle_id: int, offset: int):
    channel_elements = []
    for i, channel_name in enumerate(channel_names):
        channel_attrib = {
            "ID": "Channel:0:" + str(offset + i),
            "Name": channel_name,
            "SamplesPerPixel": "1",
        }
        channel = ET.Element("Channel", channel_attrib)
        channel_elements.append(channel)
    return channel_elements


def generate_tiffdata_meta(image_dimensions: dict):
    tiffdata_elements = []
    ifd = 0
    for t in range(0, image_dimensions["SizeT"]):
        for c in range(0, image_dimensions["SizeC"]):
            for z in range(0, image_dimensions["SizeZ"]):
                tiffdata_attrib = {
                    "FirstT": str(t),
                    "FirstC": str(c),
                    "FirstZ": str(z),
                    "IFD": str(ifd),
                }
                tiffdata = ET.Element("TiffData", tiffdata_attrib)
                tiffdata_elements.append(tiffdata)
                ifd += 1
    return tiffdata_elements


def generate_default_pixel_attributes(img_path: str) -> Dict[str, str]:
    with tif.TiffFile(img_path) as TF:
        img_dtype = TF.series[0].dtype
    pixels_attrib = {
        "ID": "Pixels:0",
        "DimensionOrder": "XYZCT",
        "Interleaved": "false",
        "Type": img_dtype.name,
    }
    return pixels_attrib


def generate_ome_meta_per_cycle(
    cycle_map: Dict[int, Dict[str, Path]],
    img_dims_per_cycle: Dict[int, Dict[str, int]],
    pixels_attrib: dict,
) -> Dict[int, XML]:
    proper_ome_attrib = {
        "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd",
    }

    channel_id_offset = 0
    channel_elements = []
    ome_xml_per_cycle = dict()
    for cyc in cycle_map:
        channel_names = list(cycle_map[cyc].keys())
        num_channels = len(channel_names)
        this_cycle_channels = generate_channel_meta(
            channel_names, cyc, channel_id_offset
        )
        #channel_elements.extend(this_cycle_channels)
        channel_id_offset += num_channels
        tiffdata_elements = generate_tiffdata_meta(img_dims_per_cycle[cyc])

        img_dims_str = {dim: str(size) for dim, size in img_dims_per_cycle[cyc].items()}

        pixels_attrib.update(img_dims_str)

        node_ome = ET.Element("OME", proper_ome_attrib)
        node_image = ET.Element("Image", {"ID": "Image:0", "Name": "default.tif"})
        node_pixels = ET.Element("Pixels", pixels_attrib)

        for ch in this_cycle_channels:
            node_pixels.append(ch)

        for td in tiffdata_elements:
            node_pixels.append(td)

        node_image.append(node_pixels)
        node_ome.append(node_image)

        xmlstr = ET.tostring(node_ome, encoding="utf-8", method="xml").decode("ascii")
        xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>'
        ome_meta = xml_declaration + xmlstr
        ome_xml = str_to_xml(ome_meta)
        ome_xml_per_cycle[cyc] = ome_xml
    return ome_xml_per_cycle


def generate_ome_for_cycle_builder(
    cycle_map: Dict[int, Dict[str, Path]]
) -> Dict[int, XML]:
    first_cycle_channels = get_first_element_of_dict(cycle_map)
    first_channel_path = list(first_cycle_channels.values())[0]

    pixels_attrib = generate_default_pixel_attributes(first_channel_path)
    image_dimensions_per_cycle = get_dimensions_per_cycle(cycle_map)
    ome_meta_per_cycle = generate_ome_meta_per_cycle(
        cycle_map, image_dimensions_per_cycle, pixels_attrib
    )
    return ome_meta_per_cycle
