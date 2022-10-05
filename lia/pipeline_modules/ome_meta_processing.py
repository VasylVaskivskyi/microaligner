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
from typing import Any, Dict, List, Tuple, Union

import tifffile as tif
import pint

from ..shared_modules.dtype_aliases import XML, Shape2D
from ..shared_modules.utils import path_to_str


def str_to_xml(xmlstr: str) -> XML:
    """Converts str to xml and strips namespaces"""
    it = ET.iterparse(StringIO(xmlstr))
    for _, el in it:
        _, _, el.tag = el.tag.rpartition("}")
    root = it.root
    return root


def read_ome_meta_from_file(path: Path) -> XML:
    with tif.TiffFile(path_to_str(path)) as TF:
        ome_meta_str = TF.ome_metadata
    ome_xml = str_to_xml(ome_meta_str)
    return ome_xml


def _convert_resolution_to_nm(value: float, unit: str) -> float:
    pint_unit = pint.UnitRegistry()
    provided_unit = pint_unit[unit]
    provided_res = value
    res_in_units = provided_res * provided_unit
    res_in_um = res_in_units.to("nm")
    return res_in_um.magnitude


def _convert_sizes(size_info: dict) -> dict:
    phys_size_x_conv = _convert_resolution_to_nm(size_info["PhysicalSizeX"], size_info["PhysicalSizeXUnit"])
    phys_size_y_conv = _convert_resolution_to_nm(size_info["PhysicalSizeY"], size_info["PhysicalSizeYUnit"])
    size_info["PhysicalSizeX"] = phys_size_x_conv
    size_info["PhysicalSizeY"] = phys_size_y_conv
    size_info["PhysicalSizeXUnit"] = "nm"
    size_info["PhysicalSizeYUnit"] = "nm"
    return size_info


def _strip_cycle_info(name) -> str:
    ch_name = re.sub(r"^(c|cyc|cycle)\d+(\s+|_|-)?", "", name)  # strip start
    ch_name2 = re.sub(r"(-\d+)?(_\d+)?$", "", ch_name)  # strip end
    return ch_name2


def _filter_ref_channel_ids(ref_ch: str, channels: List[str]) -> List[int]:
    ref_ids = []
    for _id, ch in enumerate(channels):
        if re.match(ref_ch, ch, re.IGNORECASE):
            ref_ids.append(_id)
    return ref_ids


def _find_where_ref_channel(ref_ch: str, channel_info):
    """Find if reference channel is in fluorophores or channel names and return them"""
    channel_fluors = channel_info["channel_fluors"]
    channel_names = channel_info["channel_names"]
    # strip cycle id from channel name and fluor name
    if channel_fluors != []:
        fluors = [
            _strip_cycle_info(fluor) for fluor in channel_fluors
        ]  # remove cycle name
    else:
        fluors = None
    names = [_strip_cycle_info(name) for name in channel_names]

    # check if reference channel is present somewhere
    if ref_ch in names:
        cleaned_channel_names = names
    elif fluors is not None and ref_ch in fluors:
        cleaned_channel_names = fluors
    else:
        if fluors is not None:
            msg = (
                f"Incorrect reference channel {ref_ch}. "
                + f"Available channel names: {set(names)}, fluors: {set(fluors)}"
            )
            raise ValueError(msg)
        else:
            msg = (
                f"Incorrect reference channel {ref_ch}. "
                + f"Available channel names: {set(names)}"
            )
            raise ValueError(msg)
    ref_ch_ids = _filter_ref_channel_ids(ref_ch, cleaned_channel_names)
    return cleaned_channel_names, ref_ch_ids


def collect_info_from_ome(ref_ch: str, ome_xml: XML) -> Dict[str, Any]:
    channel_info = _extract_channel_info(ome_xml)
    cleaned_channel_names, ref_ids = _find_where_ref_channel(ref_ch, channel_info)
    pixels_info = _extract_pixels_info(ome_xml)
    ome_info = channel_info.copy()
    ome_info["ref_ch_ids"] = ref_ids
    ome_info.update(pixels_info)
    return ome_info


def _extract_channel_info(ome_xml: XML) -> Dict[str, Any]:
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


def _extract_pixels_info(ome_xml: XML) -> Dict[str, Union[int, float]]:
    dims = ["SizeX", "SizeY", "SizeC", "SizeZ", "SizeT"]
    sizes = ["PhysicalSizeX", "PhysicalSizeY"]
    units = ["PhysicalSizeXUnit", "PhysicalSizeYUnit"]
    pixels = ome_xml.find("Image").find("Pixels")
    pixels_info = dict()
    for d in dims:
        pixels_info[d] = int(pixels.get(d, 1))
    for s in sizes:
        pixels_info[s] = float(pixels.get(s, 1))
    for u in units:
        pixels_info[u] = pixels.get(u, "um")
    return pixels_info


def extract_sizes_from_xml_list(
    ome_xml_list: List[XML], target_shape: Shape2D
) -> Dict[str, Union[int, Any]]:
    time = []
    planes = []
    channels = []
    phys_size_x_list = []
    phys_size_y_list = []
    for ome_xml in ome_xml_list:
        pixels_info = _extract_pixels_info(ome_xml)
        time.append(pixels_info["SizeT"])
        planes.append(pixels_info["SizeZ"])
        channels.append(pixels_info["SizeC"])
        phys_size_x_list.append(pixels_info["PhysicalSizeX"])
        phys_size_y_list.append(pixels_info["PhysicalSizeY"])
        phys_unit_x = pixels_info["PhysicalSizeXUnit"]
        phys_unit_y = pixels_info["PhysicalSizeYUnit"]
    n_time = max(time)
    n_zplanes = max(planes)
    n_channels = sum(channels)
    phys_size_x = max(phys_size_x_list)
    phys_size_y = max(phys_size_y_list)
    sizes = {
        "SizeX": target_shape[1],
        "SizeY": target_shape[0],
        "SizeC": n_channels,
        "SizeZ": n_zplanes,
        "SizeT": n_time,
        "PhysicalSizeX": phys_size_x,
        "PhysicalSizeY": phys_size_y,
        "PhysicalSizeXUnit": phys_unit_x,
        "PhysicalSizeYUnit": phys_unit_y,
    }
    return sizes


def create_channel_nodes(channel_info: Dict[str, Any]) -> List[XML]:
    channel_id = 0
    channel_nodes = []
    channels = deepcopy(channel_info["channels"])
    channel_names = channel_info["channel_names"]

    for ch in range(0, len(channels)):
        new_channel_id = "Channel:0:" + str(channel_id)
        channels[ch].set("Name", channel_names[ch])
        channels[ch].set("ID", new_channel_id)
        channel_nodes.append(channels[ch])
        channel_id += 1
    return channel_nodes


def create_channel_nodes_list(ncycles: int, channel_info_list: List[Dict[str, Any]]
) -> List[XML]:
    digit_format = (
        "0" + str(len(str(ncycles)) + 1) + "d"
    )  # e.g. for number 5 format = 02d, result = 05
    channel_id = 0
    channel_nodes = []
    for i in range(0, ncycles):
        channels = deepcopy(channel_info_list[i]["channels"])
        channel_names = channel_info_list[i]["channel_names"]

        # eg c02 DAPI
        cycle_prefix = "c" + format(i + 1, digit_format) + " "
        new_channel_names = [cycle_prefix + ch for ch in channel_names]

        for ch in range(0, len(channels)):
            channel = deepcopy(channels[ch])
            new_channel_id = "Channel:0:" + str(channel_id)
            new_channel_name = new_channel_names[ch]
            channel.set("Name", new_channel_name)
            channel.set("ID", new_channel_id)
            channel_nodes.append(channel)
            channel_id += 1
    return channel_nodes


def create_tiff_data_nodes(n_time: int, n_channels: int, n_zplanes: int) -> List[XML]:
    tiffdata_list = []
    ifd = 0
    for t in range(0, n_time):
        for c in range(0, n_channels):
            for z in range(0, n_zplanes):
                td_info = {
                    "FirstC": str(c),
                    "FirstT": str(t),
                    "FirstZ": str(z),
                    "IFD": str(ifd),
                    "PlaneCount": str(1),
                }
                td = ET.Element("TiffData", td_info)
                tiffdata_list.append(td)
                ifd += 1
    return tiffdata_list


def get_proper_ome_attribs() -> Dict[str, str]:
    proper_ome_attribs = {
        "xmlns": "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://www.openmicr oscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd",
    }
    return proper_ome_attribs


def remove_tiff_data_nodes(xml: XML) -> XML:
    new_xml = deepcopy(xml)
    tiffdata = new_xml.find("Image").find("Pixels").findall("TiffData")
    if tiffdata is not None or tiffdata != []:
        for td in tiffdata:
            new_xml.find("Image").find("Pixels").remove(td)
    return new_xml


def xml_to_string(xml: XML) -> str:
    xml_declaration = '<?xml version="1.0" encoding="UTF-8"?>'
    result_ome_meta = xml_declaration + ET.tostring(
        xml, method="xml", encoding="utf-8"
    ).decode("ascii", errors="ignore")
    return result_ome_meta


def create_meta_for_each_img(
    ome_meta_per_cyc: Dict[int, XML], target_shape: Shape2D
) -> Dict[int, str]:
    old_xmls = dict()
    ome_xml_list = []
    for cyc, ome_xml in ome_meta_per_cyc.items():
        old_xmls[cyc] = ome_xml
        ome_xml_list.append(ome_xml)

    new_xmls = dict()
    for cyc, old_xml in old_xmls.items():
        sizes = extract_sizes_from_xml_list([old_xml], target_shape)
        sizes = _convert_sizes(sizes)
        new_xml = deepcopy(old_xml)
        new_dim_order = "XYZCT"
        new_xml.find("Image").find("Pixels").set("DimensionOrder", new_dim_order)
        for attr, val in sizes.items():
            new_xml.find("Image").find("Pixels").set(attr, str(val))

        # add ome attribs
        new_xml.attrib.clear()
        proper_ome_attribs = get_proper_ome_attribs()
        for attr, val in proper_ome_attribs.items():
            new_xml.set(attr, val)

        # remove old tiffdata
        new_xml = remove_tiff_data_nodes(new_xml)

        # add tiffdata
        tiffdata_list = create_tiff_data_nodes(
            sizes["SizeT"], sizes["SizeC"], sizes["SizeZ"]
        )
        for td in tiffdata_list:
            new_xml.find("Image").find("Pixels").append(td)

        result_ome_meta = xml_to_string(new_xml)
        new_xmls[cyc] = result_ome_meta
    return new_xmls


def create_combined_meta(
    ome_meta_per_cyc: Dict[int, XML], target_shape: Shape2D
) -> Dict[int, str]:
    ncycles = len(ome_meta_per_cyc)

    ome_xml_list = []
    channel_info_list = []
    for cyc, ome_xml in ome_meta_per_cyc.items():
        channel_info = _extract_channel_info(ome_xml)
        channel_info_list.append(channel_info)
        ome_xml_list.append(ome_xml)

    sizes = extract_sizes_from_xml_list(ome_xml_list, target_shape)
    sizes = _convert_sizes(sizes)
    # use metadata from first image as reference metadata
    ref_xml = deepcopy(ome_xml_list[0])

    # set proper ome attributes tags
    ref_xml.attrib.clear()
    proper_ome_attribs = get_proper_ome_attribs()
    for attr, val in proper_ome_attribs.items():
        ref_xml.set(attr, val)

    # set new dimension sizes
    new_dim_order = "XYZCT"
    ref_xml.find("Image").find("Pixels").set("DimensionOrder", new_dim_order)
    for attr, val in sizes.items():
        ref_xml.find("Image").find("Pixels").set(attr, str(val))

    # remove old channels and tiffdata
    old_channels = ref_xml.find("Image").find("Pixels").findall("Channel")
    for ch in old_channels:
        ref_xml.find("Image").find("Pixels").remove(ch)

    # remove old tiffdata
    ref_xml = remove_tiff_data_nodes(ref_xml)

    # add new channels
    channel_info_list = create_channel_nodes_list(ncycles, channel_info_list)
    for ch in channel_info_list:
        ref_xml.find("Image").find("Pixels").append(ch)

    # add tiffdata
    tiffdata_list = create_tiff_data_nodes(
        sizes["SizeT"], sizes["SizeC"], sizes["SizeZ"]
    )
    for td in tiffdata_list:
        ref_xml.find("Image").find("Pixels").append(td)

    result_ome_meta = xml_to_string(ref_xml)

    combined_meta = dict()
    for cyc in ome_meta_per_cyc:
        combined_meta[cyc] = result_ome_meta
    return combined_meta


def separate_stack_meta(ome_meta_per_cyc: Dict[int, XML], target_shape: Shape2D
) -> Dict[int, str]:
    new_dim_order = "XYZCT"
    old_xmls = dict()
    ome_xml_list = []
    for cyc, ome_xml in ome_meta_per_cyc.items():
        old_xmls[cyc] = ome_xml
        ome_xml_list.append(ome_xml)

    sizes = extract_sizes_from_xml_list([ome_xml_list[0]], target_shape)
    ncycles = len(ome_meta_per_cyc)
    num_ch_per_cyc = int(round(sizes["SizeC"] / ncycles, 0))

    n = 0
    new_xmls = dict()
    for cyc, old_xml in old_xmls.items():
        sizes = extract_sizes_from_xml_list([old_xml], target_shape)
        sizes["SizeC"] = num_ch_per_cyc
        sizes = _convert_sizes(sizes)
        new_xml = deepcopy(old_xml)
        new_xml.find("Image").find("Pixels").set("DimensionOrder", new_dim_order)
        for attr, val in sizes.items():
            new_xml.find("Image").find("Pixels").set(attr, str(val))

        # "channels": channels,
        # "channel_names": channel_names,
        # "channel_fluors": channel_fluors,
        # "nchannels": nchannels,
        # "nzplanes": nzplanes,

        channel_info = _extract_channel_info(old_xml)
        channel_slice = slice(n * num_ch_per_cyc, (n + 1) * num_ch_per_cyc)
        channel_info["channels"] = channel_info["channels"][channel_slice]
        channel_info["channel_names"] = channel_info["channel_names"][channel_slice]
        channel_info["channel_fluors"] = channel_info["channel_fluors"][channel_slice]
        channel_info["nchannels"] = num_ch_per_cyc

        # remove old channels
        old_channels = new_xml.find("Image").find("Pixels").findall("Channel")
        for ch in old_channels:
            new_xml.find("Image").find("Pixels").remove(ch)

        # add new channels
        channel_info_list = create_channel_nodes(channel_info)
        for ch in channel_info_list:
            new_xml.find("Image").find("Pixels").append(ch)

        # add ome attribs
        new_xml.attrib.clear()
        proper_ome_attribs = get_proper_ome_attribs()
        for attr, val in proper_ome_attribs.items():
            new_xml.set(attr, val)

        # remove old tiffdata
        new_xml = remove_tiff_data_nodes(new_xml)

        # add tiffdata
        tiffdata_list = create_tiff_data_nodes(
            sizes["SizeT"], sizes["SizeC"], sizes["SizeZ"]
        )
        for td in tiffdata_list:
            new_xml.find("Image").find("Pixels").append(td)

        result_ome_meta = xml_to_string(new_xml)
        new_xmls[cyc] = result_ome_meta
        n += 1
    return new_xmls


def create_new_meta(
    ome_meta_per_cyc: Dict[int, XML],
    target_shape: Shape2D,
    input_is_stack: bool,
    output_is_stack: bool,
) -> Dict[int, str]:
    if input_is_stack and output_is_stack:
        new_ome_meta_str_per_cyc = dict()
        for cyc in ome_meta_per_cyc:
            new_ome_meta_str_per_cyc[cyc] = xml_to_string(ome_meta_per_cyc[cyc])
    elif output_is_stack:
        new_ome_meta_str_per_cyc = create_combined_meta(ome_meta_per_cyc, target_shape)
    elif input_is_stack:
        new_ome_meta_str_per_cyc = separate_stack_meta(ome_meta_per_cyc, target_shape)
    else:
        new_ome_meta_str_per_cyc = create_meta_for_each_img(ome_meta_per_cyc, target_shape)
    return new_ome_meta_str_per_cyc
