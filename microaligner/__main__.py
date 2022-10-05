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

import argparse
import gc
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

import jsonschema
import numpy as np
import pandas as pd
import tifffile as tif
import yaml

from .feature_reg import FeatureRegistrator
from .optflow_reg import OptFlowRegistrator, Warper
from .shared_modules.dtype_aliases import Flow, Padding, Shape2D, TMat
from .pipeline_modules.metadata_handling import (DatasetStruct,
                                                 DatasetStructCreator)
from .pipeline_modules.ome_meta_processing import create_new_meta
from .pipeline_modules.config_reader import PipelineConfigReader, PipelineConfig
from .shared_modules.utils import (pad_to_shape, path_to_str,
                                   read_tiff_page,
                                   read_and_max_project_pages,
                                   set_number_of_dask_workers,
                                   transform_img_with_tmat)


def get_first_element_of_dict(dictionary: dict):
    first_key = list(dictionary.keys())[0]
    return dictionary[first_key]


def save_param(
        out_dir: Path,
        tmat_per_cycle: Dict[int, TMat],
        padding_per_cycle: Dict[int, Padding],
        image_shape: Shape2D,
):
    col_names = ["name", 0, 1, 2, 3, 4, 5, "left", "right", "top", "bottom", "width", "height"]
    transform_table = pd.DataFrame(columns=col_names)

    for i, cyc in enumerate(tmat_per_cycle):
        transform_table.loc[i, [0, 1, 2, 3, 4, 5]] = tmat_per_cycle[cyc].flatten()
        transform_table.loc[i, "name"] = f"Cycle {cyc}"
        this_cycle_padding = padding_per_cycle[cyc]
        transform_table.loc[i, "left"] = this_cycle_padding[0]
        transform_table.loc[i, "right"] = this_cycle_padding[1]
        transform_table.loc[i, "top"] = this_cycle_padding[2]
        transform_table.loc[i, "bottom"] = this_cycle_padding[3]
        transform_table.loc[i, "width"] = image_shape[1]
        transform_table.loc[i, "height"] = image_shape[0]
    transform_table.to_csv(out_dir / "feature_reg_parameters.csv", index=False)


def transform_and_save_zplanes(
        mm: tif.memmap,
        ch_id: int,
        target_shape: Shape2D,
        transform_matrix: TMat,
        img_paths: Dict[int, Path],
        tiff_pages: Dict[int, int],
        max_zplanes: int,
):
    z_id = 0
    for z, img_path in img_paths.items():
        img = read_tiff_page(img_path, tiff_pages[z])
        #img = tif.imread(path_to_str(img_path), key=tiff_pages[z])

        img = transform_img_with_tmat(img, target_shape, transform_matrix)
        mm[0, ch_id, z_id, :, :] = img
        mm.flush()
        gc.collect()
        z_id += 1

    num_zplanes = len(tiff_pages)
    if num_zplanes < max_zplanes:
        diff = max_zplanes - num_zplanes
        empty_page = np.zeros_like(img)
        for a in range(0, diff):
            mm[0, ch_id, num_zplanes + a, :, :] = empty_page
            mm.flush()
        del empty_page
    gc.collect()
    del img
    return


def create_memmap_for_saving(output_path: Path,
                            img_shape: Tuple[int, int, int, int, int],
                            img_dtype: np.dtype,
                            ome_meta: str
) -> tif.memmap:
    mm = tif.memmap(output_path,
                    shape=img_shape,
                    mode="r+",
                    dtype=img_dtype,
                    photometric="minisblack",
                    bigtiff=True,
                    description=ome_meta,
                    contiguous=True
                    )
    return mm


def transform_and_save_freg_imgs(
        dataset_struct: DatasetStruct,
        out_dir: Path,
        filenames: Dict[str, str],
        target_shape: Shape2D,
        tmat_per_cycle: Dict[int, TMat],
        ome_meta_per_cyc: Dict[int, str],
        input_is_stack: bool,
        save_to_stack: bool,
):
    print("Transforming images")

    cycles = sorted(list(dataset_struct.tiff_pages.keys()))
    first_cycle = cycles[0]
    ncycles = len(cycles)

    first_cycle_paths = dataset_struct.img_paths[first_cycle]
    zplane_paths = get_first_element_of_dict(first_cycle_paths)
    img_path = get_first_element_of_dict(zplane_paths)
    with tif.TiffFile(path_to_str(img_path)) as TF:
        img_dtype = TF.series[0].dtype

    nzplanes_per_cyc = []
    for cyc in dataset_struct.tiff_pages:
        for ch in dataset_struct.tiff_pages[cyc]:
            nzplanes_per_cyc.append(len(dataset_struct.tiff_pages[cyc][ch]))

    nchannels_per_cyc = []
    for cyc in dataset_struct.tiff_pages:
        nchannels_per_cyc.append(len(dataset_struct.tiff_pages[cyc]))

    max_zplanes = max(nzplanes_per_cyc)
    total_channels = sum(nchannels_per_cyc)
    if save_to_stack:
        output_path = out_dir / filenames["stack"]
        ome_meta = ome_meta_per_cyc[first_cycle]
        #TW = tif.TiffWriter(output_path, bigtiff=True)
        cycle_stack_shape = (1, total_channels, max_zplanes, target_shape[0], target_shape[1])
        img_memmap = create_memmap_for_saving(output_path, cycle_stack_shape, img_dtype, ome_meta)

    for cyc_id, cyc in enumerate(dataset_struct.tiff_pages):
        print(f"Transforming and saving Cycle {cyc} [{cyc_id + 1}/{ncycles}]")
        if not save_to_stack:
            filename = filenames["per_cycle"].format(cyc=cyc)
            cyc_out_path = out_dir / filename
            #TW = tif.TiffWriter(cyc_out_path, bigtiff=True)
            ome_meta = ome_meta_per_cyc[cyc]
            nchannels = len(dataset_struct.tiff_pages[cyc])
            stack_shape = (1, nchannels, max_zplanes, target_shape[0], target_shape[1])
            img_memmap = create_memmap_for_saving(cyc_out_path, stack_shape, img_dtype, ome_meta)

        transform_matrix = tmat_per_cycle[cyc]

        for ch_id, ch in enumerate(dataset_struct.tiff_pages[cyc]):
            cross_cyc_ch_id = cyc_id * nchannels_per_cyc[0] + ch_id
            tiff_pages = dataset_struct.tiff_pages[cyc][ch]
            img_paths = dataset_struct.img_paths[cyc][ch]
            transform_and_save_zplanes(
                img_memmap,
                cross_cyc_ch_id,
                target_shape,
                transform_matrix,
                img_paths,
                tiff_pages,
                max_zplanes
            )
        if not save_to_stack:
            del img_memmap
    if save_to_stack:
        del img_memmap
    return


def get_target_shape(img_paths: List[Path]) -> Shape2D:
    img_shapes = []
    for i in range(0, len(img_paths)):
        with tif.TiffFile(img_paths[i]) as TF:
            img_axes = TF.series[0].axes
            y_ax = img_axes.index("Y")
            x_ax = img_axes.index("X")
            img_shape = TF.series[0].shape
            img_shapes.append((img_shape[y_ax], img_shape[x_ax]))
    max_size_x = max([s[1] for s in img_shapes])
    max_size_y = max([s[0] for s in img_shapes])
    target_shape = (max_size_y, max_size_x)
    return target_shape


def do_feature_reg(
        dataset_struct: DatasetStruct,
        ref_cycle_id: int,
        num_pyr_lvl: int,
        num_iter: int,
        tile_size: int,
        target_shape: Shape2D,
        use_full_res_img: bool,
        use_dog: bool,
) -> Tuple[Dict[int, TMat], Dict[int, Padding]]:
    freg = FeatureRegistrator()
    freg.num_pyr_lvl = num_pyr_lvl
    freg.num_iterations = num_iter
    freg.tile_size = tile_size
    freg.use_full_res_img = use_full_res_img
    freg.use_dog = use_dog

    tmat_per_cycle = dict()
    padding = dict()
    identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    ref_channel_id = dataset_struct.ref_channel_ids[ref_cycle_id]
    tiff_pages = dataset_struct.tiff_pages[ref_cycle_id][ref_channel_id]
    img_paths = dataset_struct.img_paths[ref_cycle_id][ref_channel_id]
    ref_img = read_and_max_project_pages(img_paths, tiff_pages)
    ref_img, ref_pad = pad_to_shape(ref_img, target_shape)
    freg.ref_img = ref_img

    cycles = list(dataset_struct.tiff_pages.keys())
    ncycles = len(cycles)
    for cyc_id, cyc in enumerate(cycles):
        print(f"Processing Cycle {cyc} [{cyc_id + 1}/{ncycles}]")
        if cyc == ref_cycle_id:
            print("Skipping as it is a reference cycle")
            tmat_per_cycle[cyc] = identity_matrix
            padding[cyc] = ref_pad
        else:
            ref_channel_id = dataset_struct.ref_channel_ids[cyc]
            tiff_pages = dataset_struct.tiff_pages[cyc][ref_channel_id]
            img_paths = dataset_struct.img_paths[cyc][ref_channel_id]

            mov_img = read_and_max_project_pages(img_paths, tiff_pages)
            gc.collect()

            mov_img, pad = pad_to_shape(mov_img, target_shape)
            padding[cyc] = pad

            freg.mov_img = mov_img
            transform_matrix = freg.register(reuse_ref_img=True)

            tmat_per_cycle[cyc] = transform_matrix
            gc.collect()
    return tmat_per_cycle, padding


def warp_and_save_pages(
        mm: tif.memmap,
        ch_id: int,
        warper: Warper,
        flow: Flow,
        img_paths: Dict[int, Path],
        tiff_pages: Dict[int, int],
):
    for z_id, z in enumerate(img_paths):
        warper.image = read_tiff_page(img_paths[z], tiff_pages[z])
        warper.flow = flow
        warped_img = warper.warp()
        mm[0, ch_id, z_id, :, :] = warped_img
        mm.flush()
    return


def save_pages(
        mm: tif.memmap,
        ch_id: int,
        img_paths: Dict[int, Path],
        tiff_pages: Dict[int, int],
):
    for z_id, z in enumerate(img_paths):
        #img = read_tiff_page(img_paths[z], tiff_pages[z])
        mm[0, ch_id, z_id, :, :] = read_tiff_page(img_paths[z], tiff_pages[z])  # tif.imread(path_to_str(img_paths[z]), key=tiff_pages[z])
        mm.flush()
    return


def register_and_save_ofreg_imgs(
        dataset_struct: DatasetStruct,
        out_dir: Path,
        filenames: Dict[str, str],
        tile_size: int,
        overlap: int,
        num_pyr_lvl: int,
        num_iter: int,
        ome_meta_per_cyc: Dict[int, str],
        input_is_stack: bool,
        save_to_stack: bool,
        use_full_res_img: bool,
        use_dog: bool,
):
    """Read images and register them sequentially: 1<-2, 2<-3, 3<-4 etc.
    It is assumed that there is equal number of channels in each cycle.
    """
    ofreg = OptFlowRegistrator()
    ofreg.tile_size = tile_size
    ofreg.overlap = overlap
    ofreg.num_pyr_lvl = num_pyr_lvl
    ofreg.num_iterations = num_iter
    ofreg.use_full_res_img = use_full_res_img
    ofreg.use_dog = use_dog

    warper = Warper()
    warper.tile_size = tile_size
    warper.overlap = overlap

    cycles = list(dataset_struct.tiff_pages.keys())
    first_cycle = cycles[0]
    ncycles = len(cycles)

    first_cycle_paths = dataset_struct.img_paths[first_cycle]
    zplane_paths = get_first_element_of_dict(first_cycle_paths)
    img_path = get_first_element_of_dict(zplane_paths)
    with tif.TiffFile(path_to_str(img_path)) as TF:
        img_shape = TF.series[0].shape
        img_dtype = TF.series[0].dtype

    nzplanes_per_cyc = []
    for cyc in dataset_struct.tiff_pages:
        for ch in dataset_struct.tiff_pages[cyc]:
            nzplanes_per_cyc.append(len(dataset_struct.tiff_pages[cyc][ch]))

    nchannels_per_cyc = []
    for cyc in dataset_struct.tiff_pages:
        nchannels_per_cyc.append(len(dataset_struct.tiff_pages[cyc]))

    max_zplanes = max(nzplanes_per_cyc)
    total_channels = sum(nchannels_per_cyc)

    if save_to_stack:
        ome_meta = ome_meta_per_cyc[first_cycle]
        out_path = out_dir / filenames["stack"]
        cycle_stack_shape = (1, total_channels, max_zplanes, img_shape[-2], img_shape[-1])
        img_memmap = create_memmap_for_saving(out_path, cycle_stack_shape, img_dtype, ome_meta)
        #TW = tif.TiffWriter(out_path, bigtiff=True)

    for cyc_id, cyc in enumerate(cycles):
        print(f"Processing Cycle {cyc} [{cyc_id + 1}/{ncycles}]")
        if not save_to_stack:
            ome_meta = ome_meta_per_cyc[cyc]
            filename = filenames["per_cycle"].format(cyc=cyc)
            cyc_out_path = out_dir / filename
            #TW = tif.TiffWriter(cyc_out_path, bigtiff=True)
            nchannels = len(dataset_struct.tiff_pages[cyc])
            stack_shape = (1, nchannels, max_zplanes, img_shape[-2], img_shape[-1])
            img_memmap = create_memmap_for_saving(cyc_out_path, stack_shape, img_dtype, ome_meta)

        ref_ch_id = dataset_struct.ref_channel_ids[cyc]
        img_paths: Dict[int, Path] = dataset_struct.img_paths[cyc][ref_ch_id]
        tiff_pages: Dict[int, int] = dataset_struct.tiff_pages[cyc][ref_ch_id]

        if cyc == first_cycle:
            print("Skipping as it is a reference image")
            ref_img = read_and_max_project_pages(img_paths, tiff_pages)

            print(f"Saving Cycle {cyc} [{cyc_id + 1}/{ncycles}]")
            for ch_id, ch in enumerate(dataset_struct.tiff_pages[cyc]):
                tiff_pages = dataset_struct.tiff_pages[cyc][ch]
                img_paths = dataset_struct.img_paths[cyc][ch]
                cross_cyc_ch_id = cyc_id * nchannels_per_cyc[0] + ch_id
                save_pages(img_memmap, cross_cyc_ch_id, img_paths, tiff_pages)
        else:
            # mov_pages = list(this_cycle["img_structure"][ref_ch_id].values())
            mov_img = read_and_max_project_pages(img_paths, tiff_pages)

            ofreg.ref_img = ref_img  # comes from previous cycle
            ofreg.mov_img = mov_img
            flow = ofreg.register()

            warper.image = mov_img
            warper.flow = flow
            ref_img = warper.warp()  # will be used in the next cycle

            print(f"Saving Cycle {cyc} [{cyc_id + 1}/{ncycles}]")
            for ch_id, ch in enumerate(dataset_struct.tiff_pages[cyc]):
                tiff_pages = dataset_struct.tiff_pages[cyc][ch]
                img_paths = dataset_struct.img_paths[cyc][ch]
                cross_cyc_ch_id = cyc_id * nchannels_per_cyc[0] + ch_id
                warp_and_save_pages(img_memmap, cross_cyc_ch_id, warper, flow, img_paths, tiff_pages)
        if not save_to_stack:
            del img_memmap
    if save_to_stack:
        del img_memmap


def parse_cmd_args() -> Path:
    parser = argparse.ArgumentParser(description="MicroAligner: image registration for large scale microscopy")
    parser.add_argument("config", type=Path, help="path to the config yaml file")
    args = parser.parse_args()
    reg_config_path = args.config
    return reg_config_path


def run_feature_reg(config: PipelineConfig, target_shape: Shape2D):
    print("Performing linear feature based image registration")

    img_paths = config.Input.InputImagePaths
    input_is_stack = config.Input.PipelineInputType == "CycleStack"
    input_is_stack_builder = config.Input.PipelineInputType == "CycleBuilder"
    output_is_stack = config.Output.SaveOutputToCycleStack
    out_dir = Path(config.Output.OutputDir)
    out_prefix = config.Output.OutputPrefix
    ref_cycle_id = config.Input.ReferenceCycle
    ref_ch = config.Input.ReferenceChannel

    freg_reg_param = config.RegistrationParameters.FeatureReg
    n_workers = freg_reg_param.NumberOfWorkers
    num_pyr_lvl = freg_reg_param.NumberPyramidLevels
    num_iter = freg_reg_param.NumberIterationsPerLevel
    tile_size = freg_reg_param.TileSize
    use_full_res_img = freg_reg_param.UseFullResImage
    use_dog = freg_reg_param.UseDOG

    set_number_of_dask_workers(n_workers)
    struct = DatasetStructCreator()
    struct.img_paths = img_paths
    struct.input_is_stack = input_is_stack
    struct.input_is_stack_builder = input_is_stack_builder
    struct.output_is_stack = output_is_stack
    struct.ref_channel_name = ref_ch
    dataset_struct = struct.create_dataset_struct()

    new_ome_meta = create_new_meta(
        dataset_struct.ome_xmls, target_shape, input_is_stack, output_is_stack
    )

    tmat_per_cycle, padding_per_cycle = do_feature_reg(
        dataset_struct,
        ref_cycle_id,
        num_pyr_lvl,
        num_iter,
        tile_size,
        target_shape,
        use_full_res_img,
        use_dog,
    )

    feature_reg_out_file_names = {
        "stack": out_prefix + "feature_reg_result_stack.tif",
        "per_cycle": out_prefix + "feature_reg_result_cyc{cyc:03d}.tif",
    }
    transform_and_save_freg_imgs(
        dataset_struct,
        out_dir,
        feature_reg_out_file_names,
        target_shape,
        tmat_per_cycle,
        new_ome_meta,
        input_is_stack,
        output_is_stack,
    )
    save_param(
        out_dir, tmat_per_cycle, padding_per_cycle, target_shape
    )
    if output_is_stack:
        img_paths = {"CycleStack": out_dir / feature_reg_out_file_names["stack"]}
    else:
        img_paths = dict()
        for cyc in dataset_struct.img_paths:
            filename = feature_reg_out_file_names["per_cycle"].format(cyc=cyc)
            img_paths[cyc] = out_dir / filename
    print("Finished\n")
    return img_paths


def check_input_img_dims_match(img_paths: List[Path]) -> bool:
    img_shapes = []
    for i in range(0, len(img_paths)):
        with tif.TiffFile(img_paths[i]) as TF:
            img_axes = TF.series[0].axes
            y_ax = img_axes.index("Y")
            x_ax = img_axes.index("X")
            img_shape = TF.series[0].shape
            img_shapes.append((img_shape[y_ax], img_shape[x_ax]))
    img_shapes_match = [img_shapes[0] == sh for sh in img_shapes]
    all_match = all(img_shapes_match)
    return all_match


def run_opt_flow_reg(
        config, img_paths: List[Path], target_shape: Shape2D
):
    input_is_stack = config.Input.PipelineInputType == "CycleStack"
    input_is_stack_builder = config.Input.PipelineInputType == "CycleBuilder"
    output_is_stack = config.Output.SaveOutputToCycleStack
    out_dir = Path(config.Output.OutputDir)
    out_prefix = config.Output.OutputPrefix
    ref_cycle_id = config.Input.ReferenceCycle
    ref_ch = config.Input.ReferenceChannel

    optflow_reg_param = config.RegistrationParameters.OptFlowReg
    n_workers = optflow_reg_param.NumberOfWorkers
    num_pyr_lvl = optflow_reg_param.NumberPyramidLevels
    num_iter = optflow_reg_param.NumberIterationsPerLevel
    tile_size = optflow_reg_param.TileSize
    overlap = optflow_reg_param.Overlap
    use_full_res_img = optflow_reg_param.UseFullResImage
    use_dog = optflow_reg_param.UseDOG

    need_to_run_freg = False
    if config.RegistrationParameters.FeatureReg is not None:
        input_is_stack_of = output_is_stack
        input_is_stack_builder = False
    else:
        input_is_stack_of = input_is_stack
        img_path_list = [Path(path) for key, path in config.Input.InputImagePaths.items()]
        if not input_is_stack_of:
            dims_match = check_input_img_dims_match(img_path_list)
            if not dims_match:
                print(
                    "Image dimensions do not match. "
                    + "This probably means that they are not aligned. "
                    + "Will try to perform FeatureReg first"
                )
                config.RegistrationParameters.FeatureReg = optflow_reg_param
                need_to_run_freg = True

    if need_to_run_freg:
        img_paths = run_feature_reg(config, target_shape)
        input_is_stack_of = output_is_stack

    set_number_of_dask_workers(n_workers)

    struct = DatasetStructCreator()
    struct.img_paths = img_paths
    struct.input_is_stack = input_is_stack_of
    struct.input_is_stack_builder = input_is_stack_builder
    struct.output_is_stack = output_is_stack
    struct.ref_channel_name = ref_ch
    new_dataset_struct = struct.create_dataset_struct()

    new_ome_meta = create_new_meta(
        new_dataset_struct.ome_xmls, target_shape, input_is_stack_of, output_is_stack
    )
    optflow_reg_out_file_names = {
        "stack": out_prefix + "optflow_reg_result_stack.tif",
        "per_cycle": out_prefix + "optflow_reg_result_cyc{cyc:03d}.tif",
    }
    print("Performing non-linear optical flow based image registration")
    register_and_save_ofreg_imgs(
        new_dataset_struct,
        out_dir,
        optflow_reg_out_file_names,
        tile_size,
        overlap,
        num_pyr_lvl,
        num_iter,
        new_ome_meta,
        input_is_stack,
        output_is_stack,
        use_full_res_img,
        use_dog,
    )
    print("Finished\n")
    return


def get_img_path_list(config: PipelineConfig) -> List[Path]:
    img_paths = []
    if config.Input.PipelineInputType == "CycleBuilder":
        cycle_map_raw = config.Input.InputImagePaths
        for cyc in cycle_map_raw:
            for ch in cycle_map_raw[cyc]:
                img_paths.append(cycle_map_raw[cyc][ch])
    else:
        img_paths = [Path(path) for cyc, path in config.Input.InputImagePaths.items()]
    return img_paths


def main():
    print("Started\n")
    config_path = parse_cmd_args()
    reader = PipelineConfigReader()
    config = reader.read_config(config_path)
    print("The input config is:")
    pprint(config, sort_dicts=False, indent=2)

    if not config.Output.OutputDir.exists():
        config.Output.OutputDir.mkdir(parents=True)
    img_path_list = get_img_path_list(config)
    target_shape = get_target_shape(img_path_list)

    img_paths = config.Input.InputImagePaths
    if config.RegistrationParameters.FeatureReg is not None:
        img_paths = run_feature_reg(config, target_shape)

    if config.RegistrationParameters.OptFlowReg is not None:
        run_opt_flow_reg(config, img_paths, target_shape)


if __name__ == "__main__":
    main()
