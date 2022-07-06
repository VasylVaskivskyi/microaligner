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

import argparse
import gc
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from pprint import pprint

import jsonschema
import numpy as np
import pandas as pd
import tifffile as tif
import yaml

from .feature_reg import FeatureRegistrator
from .optflow_reg import OptFlowRegistrator, Warper
from .shared_modules.config_schema_container import config_schema
from .shared_modules.dtype_aliases import Flow, Image, Padding, Shape2D, TMat
from .shared_modules.img_checks import check_number_of_input_img_paths
from .shared_modules.metadata_handling import DatasetStructure
from .shared_modules.utils import (pad_to_shape, path_to_str,
                                   read_and_max_project_pages,
                                   set_number_of_dask_workers,
                                   transform_img_with_tmat)


def save_param(
    img_paths: List[Path],
    out_dir: Path,
    tmat_per_cycle_flat: List[TMat],
    padding_per_cycle: List[Padding],
    image_shape: Shape2D,
):
    transform_table = pd.DataFrame(tmat_per_cycle_flat)
    for i in transform_table.index:
        dataset_name = "dataset_{id}_{name}".format(id=i + 1, name=img_paths[i].parent)
        transform_table.loc[i, "name"] = dataset_name
    cols = transform_table.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    transform_table = transform_table[cols]
    for i in range(0, len(padding_per_cycle)):
        this_cycle_padding = padding_per_cycle[i]
        transform_table.loc[i, "left"] = this_cycle_padding[0]
        transform_table.loc[i, "right"] = this_cycle_padding[1]
        transform_table.loc[i, "top"] = this_cycle_padding[2]
        transform_table.loc[i, "bottom"] = this_cycle_padding[3]
        transform_table.loc[i, "width"] = image_shape[1]
        transform_table.loc[i, "height"] = image_shape[0]
    transform_table.to_csv(out_dir / "feature_reg_parameters.csv", index=False)


def transform_and_save_zplanes(
    reader: tif.TiffReader,
    writer: tif.TiffWriter,
    target_shape: Shape2D,
    transform_matrix: TMat,
    zplane_pages: List[int],
    max_zplanes: int,
    ome_meta: str,
):
    for p in zplane_pages:
        img = reader.asarray(key=p)

        img = transform_img_with_tmat(img, target_shape, transform_matrix)
        writer.write(
            img, contiguous=True, photometric="minisblack", description=ome_meta
        )
        gc.collect()
    num_zplanes = len(zplane_pages)
    if num_zplanes < max_zplanes:
        diff = max_zplanes - num_zplanes
        empty_page = np.zeros_like(img)
        for a in range(0, diff):
            writer.write(
                empty_page,
                contiguous=True,
                photometric="minisblack",
                description=ome_meta,
            )
        del empty_page
    gc.collect()
    del img
    return


def transform_and_save_freg_imgs(
    dataset_structure: Dict[Any, Any],
    out_dir: Path,
    filenames: Dict[str, str],
    target_shape: Shape2D,
    transform_matrices: List[TMat],
    ome_meta_per_cyc: Dict[Path, str],
    input_is_stack: bool,
    save_to_stack: bool,
):
    print("Transforming images")
    input_img_paths = [dataset_structure[cyc]["img_path"] for cyc in dataset_structure]

    if input_is_stack:
        with tif.TiffFile(path_to_str(input_img_paths[0])) as TF:
            ome_meta = TF.ome_metadata

    ncycles = len(dataset_structure.keys())
    nzplanes = {
        cyc: len(dataset_structure[cyc]["img_structure"][0].keys())
        for cyc in dataset_structure
    }
    max_zplanes = max(nzplanes.values())
    if save_to_stack:
        output_path = out_dir / filenames["stack"]
        TW = tif.TiffWriter(output_path, bigtiff=True)
        ome_meta = ome_meta_per_cyc[Path("combined")]

    for cyc in dataset_structure:
        print(f"Transforming and saving image {cyc + 1}/{ncycles}")
        img_path = dataset_structure[cyc]["img_path"]

        if not save_to_stack:
            filename = filenames["per_cycle"].format(cyc=cyc + 1)
            cyc_out_path = out_dir / filename
            TW = tif.TiffWriter(cyc_out_path, bigtiff=True)
            ome_meta = ome_meta_per_cyc[img_path]

        TF = tif.TiffFile(img_path)
        transform_matrix = transform_matrices[cyc]

        img_structure = dataset_structure[cyc]["img_structure"]
        for channel in img_structure:
            zplane_pages = list(img_structure[channel].values())
            transform_and_save_zplanes(
                TF,
                TW,
                target_shape,
                transform_matrix,
                zplane_pages,
                max_zplanes,
                ome_meta,
            )
        TF.close()
        if not save_to_stack:
            TW.close()
    if save_to_stack:
        TW.close()
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
    dataset_structure: dict,
    ref_cycle_id: int,
    num_pyr_lvl: int,
    num_iter: int,
    tile_size: int,
    target_shape: Shape2D,
    use_full_res_img: bool,
    use_dog: bool,
) -> Tuple[List[TMat], List[Padding]]:
    freg = FeatureRegistrator()
    freg.num_pyr_lvl = num_pyr_lvl
    freg.num_iterations = num_iter
    freg.tile_size = tile_size
    freg.use_full_res_img = use_full_res_img
    freg.use_dog = use_dog

    tmat_per_cycle = []
    padding = []
    identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ncycles = len(dataset_structure)

    ref_channel_id = dataset_structure[ref_cycle_id]["ref_channel_id"]
    img_path = dataset_structure[ref_cycle_id]["img_path"]
    ref_pages = list(
        dataset_structure[ref_cycle_id]["img_structure"][ref_channel_id].values()
    )
    freg.ref_img = read_and_max_project_pages(img_path, ref_pages)

    for cyc in dataset_structure:
        print(f"Processing image {cyc + 1}/{ncycles}")
        img_structure = dataset_structure[cyc]["img_structure"]
        ref_channel_id = dataset_structure[cyc]["ref_channel_id"]
        img_path = dataset_structure[cyc]["img_path"]

        if cyc == ref_cycle_id:
            print("Skipping as it is a reference image")
            tmat_per_cycle.append(identity_matrix)
            padding.append((0, 0, 0, 0))
        else:
            mov_img_tiff_pages = list(img_structure[ref_channel_id].values())
            mov_img = read_and_max_project_pages(img_path, mov_img_tiff_pages)
            gc.collect()

            mov_img, pad = pad_to_shape(mov_img, target_shape)
            padding.append(pad)

            freg.mov_img = mov_img
            transform_matrix = freg.register(reuse_ref_img=True)

            tmat_per_cycle.append(transform_matrix)
            gc.collect()
    return tmat_per_cycle, padding


def warp_and_save_pages(
    writer: tif.TiffWriter,
    flow: Flow,
    in_path: Path,
    meta: str,
    pages: List[int],
    warper: Warper,
):
    for p in pages:
        warper.image = tif.imread(in_path, key=p)
        warper.flow = flow
        warped_img = warper.warp()
        writer.write(
            warped_img,
            contiguous=True,
            photometric="minisblack",
            description=meta,
        )


def save_pages(writer: tif.TiffWriter, in_path: Path, meta: str, pages: List[int]):
    for p in pages:
        writer.write(
            tif.imread(in_path, key=p),
            contiguous=True,
            photometric="minisblack",
            description=meta,
        )


def register_and_save_ofreg_imgs(
    dataset_structure: dict,
    out_dir: Path,
    filenames: Dict[str, str],
    tile_size: int,
    overlap: int,
    num_pyr_lvl: int,
    num_iter: int,
    ome_meta_per_cyc: Dict[Path, str],
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

    input_img_paths = [dataset_structure[cyc]["img_path"] for cyc in dataset_structure]

    if input_is_stack:
        with tif.TiffFile(path_to_str(input_img_paths[0])) as TF:
            ome_meta = TF.ome_metadata

    if save_to_stack:
        ome_meta = ome_meta_per_cyc[Path("combined")]
        out_path = out_dir / filenames["stack"]
        TW = tif.TiffWriter(out_path, bigtiff=True)

    ncycles = len(dataset_structure)

    for cyc in dataset_structure:
        print(f"Processing image {cyc + 1}/{ncycles}")
        this_cycle = dataset_structure[cyc]
        img_path = this_cycle["img_path"]
        ref_ch_id = this_cycle["ref_channel_id"]
        if not save_to_stack:
            img_path = dataset_structure[cyc]["img_path"]
            ome_meta = ome_meta_per_cyc[img_path]

            filename = filenames["per_cycle"].format(cyc=cyc + 1)
            cyc_out_path = out_dir / filename
            TW = tif.TiffWriter(cyc_out_path, bigtiff=True)

        if cyc == 0:
            print("Skipping as it is a reference image")
            ref_pages = list(this_cycle["img_structure"][ref_ch_id].values())
            ref_img = read_and_max_project_pages(img_path, ref_pages)

            print(f"Saving image {cyc + 1}/{ncycles}")
            for ch in this_cycle["img_structure"]:
                pages = list(this_cycle["img_structure"][ch].values())
                save_pages(TW, img_path, ome_meta, pages)
        else:
            mov_pages = list(this_cycle["img_structure"][ref_ch_id].values())
            mov_img = read_and_max_project_pages(img_path, mov_pages)

            ofreg.ref_img = ref_img  # comes from previous cycle
            ofreg.mov_img = mov_img
            flow = ofreg.register()

            warper.image = mov_img
            warper.flow = flow
            ref_img = warper.warp()  # will be used in the next cycle

            print(f"Saving image {cyc + 1}/{ncycles}")
            for ch in this_cycle["img_structure"]:
                pages = list(this_cycle["img_structure"][ch].values())
                warp_and_save_pages(TW, flow, img_path, ome_meta, pages, warper)
        if not save_to_stack:
            TW.close()
    if save_to_stack:
        TW.close()


def parse_cmd_args() -> Path:
    parser = argparse.ArgumentParser(description="LIA: Large image aligner")
    parser.add_argument("config", type=Path, help="path to the config yaml file")
    args = parser.parse_args()
    reg_config_path = args.config
    return reg_config_path


def validate_config(config_path: Path):
    if not config_path.exists():
        msg = "Registration config file is not found"
        raise FileNotFoundError(msg)
    config = read_yaml(config_path)
    jsonschema.validate(config, config_schema)
    missing_imgs = []
    for img_path in config["Input"]["InputImagePaths"]:
        if not Path(img_path).exists():
            missing_imgs.append(img_path)
    if missing_imgs != []:
        msg = f"These input images are not found: {missing_imgs}"
        raise FileNotFoundError(msg)
    check_number_of_input_img_paths(
        config["Input"]["InputImagePaths"], config["Input"]["InputIsCycleStack"]
    )
    return


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as s:
        yaml_file = yaml.safe_load(s)
    return yaml_file


def run_feature_reg(config, target_shape):
    print("Performing linear feature based image registration")

    img_paths = [Path(p) for p in config["Input"]["InputImagePaths"]]
    input_is_stack = config["Input"]["InputIsCycleStack"]
    save_to_stack = config["Output"]["SaveOutputToCycleStack"]
    out_dir = Path(config["Output"]["OutputDir"])
    ref_cycle_id = config["DataStructure"]["ReferenceImage"]

    freg_reg_param = config["RegistrationParameters"]["FeatureReg"]
    n_workers = freg_reg_param["NumberOfWorkers"]
    num_pyr_lvl = freg_reg_param["NumberPyramidLevels"]
    num_iter = freg_reg_param["NumberIterationsPerLevel"]
    tile_size = freg_reg_param["TileSize"]
    use_full_res_img = freg_reg_param["UseFullResImage"]
    use_dog = freg_reg_param["UseDOG"]

    set_number_of_dask_workers(n_workers)
    struct = DatasetStructure()
    struct.img_paths = img_paths
    struct.is_stack = input_is_stack
    struct.output_is_stack = save_to_stack
    dataset_structure = struct.get_dataset_structure()
    tmat_per_cycle, padding_per_cycle = do_feature_reg(
        dataset_structure,
        ref_cycle_id,
        num_pyr_lvl,
        num_iter,
        tile_size,
        target_shape,
        use_full_res_img,
        use_dog
    )
    new_ome_meta = struct.generate_new_metadata(target_shape)
    feature_reg_out_file_names = {
        "stack": "feature_reg_result_stack.tif",
        "per_cycle": "feature_reg_result_cyc{cyc:03d}.tif",
    }
    transform_and_save_freg_imgs(
        dataset_structure,
        out_dir,
        feature_reg_out_file_names,
        target_shape,
        tmat_per_cycle,
        new_ome_meta,
        input_is_stack,
        save_to_stack,
    )
    tmat_per_cycle_flat = [M.flatten() for M in tmat_per_cycle]
    img_paths = [dataset_structure[cyc]["img_path"] for cyc in dataset_structure]
    save_param(img_paths, out_dir, tmat_per_cycle_flat, padding_per_cycle, target_shape)
    if save_to_stack:
        img_paths = [out_dir / feature_reg_out_file_names["stack"]]
    else:
        img_paths = []
        for cyc in dataset_structure:
            filename = feature_reg_out_file_names["per_cycle"].format(cyc=cyc + 1)
            img_paths.append(out_dir / filename)
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
    config, feature_reg: str, img_paths: List[Path], target_shape: Shape2D
):
    input_is_stack = config["Input"]["InputIsCycleStack"]
    save_to_stack = config["Output"]["SaveOutputToCycleStack"]
    out_dir = Path(config["Output"]["OutputDir"])
    ref_cycle_id = config["DataStructure"]["ReferenceImage"]

    optflow_reg_param = config["RegistrationParameters"]["OptFlowReg"]
    n_workers = optflow_reg_param["NumberOfWorkers"]
    num_pyr_lvl = optflow_reg_param["NumberPyramidLevels"]
    num_iter = optflow_reg_param["NumberIterationsPerLevel"]
    tile_size = optflow_reg_param["TileSize"]
    overlap = optflow_reg_param["Overlap"]
    use_full_res_img = optflow_reg_param["UseFullResImage"]
    use_dog = optflow_reg_param["UseDOG"]

    need_to_run_freg = False
    if feature_reg in config["RegistrationParameters"]:
        input_is_stack_of = save_to_stack
        img_paths = img_paths
        dims_match = check_input_img_dims_match(img_paths)
        if not dims_match:
            msg = f"Something went wrong because images have different shape after performing {feature_reg}"
            raise Exception(msg)
    else:
        input_is_stack_of = input_is_stack
        img_paths = [Path(p) for p in config["Input"]["InputImagePaths"]]
        dims_match = check_input_img_dims_match(img_paths)
        if not dims_match:
            print(
                "Image dimensions do not match. "
                + "This probably means that they are not aligned. "
                + "Will try to perform FeatureReg first"
            )
            config["RegistrationParameters"][feature_reg] = optflow_reg_param
            need_to_run_freg = True

    if need_to_run_freg:
        img_paths = run_feature_reg(config, target_shape)
        input_is_stack_of = save_to_stack

    set_number_of_dask_workers(n_workers)

    struct = DatasetStructure()
    struct.img_paths = img_paths
    struct.input_is_stack = input_is_stack_of
    struct.output_is_stack = save_to_stack
    new_dataset_structure = struct.get_dataset_structure()
    new_ome_meta = struct.generate_new_metadata(target_shape)
    optflow_reg_out_file_names = {
        "stack": "optflow_reg_result_stack.tif",
        "per_cycle": "optflow_reg_result_cyc{cyc:03d}.tif",
    }
    print("Performing non-linear optical flow based image registration")
    register_and_save_ofreg_imgs(
        new_dataset_structure,
        out_dir,
        optflow_reg_out_file_names,
        tile_size,
        overlap,
        num_pyr_lvl,
        num_iter,
        new_ome_meta,
        input_is_stack,
        save_to_stack,
        use_full_res_img,
        use_dog
    )
    print("Finished\n")
    return


def main():
    print("Started\n")
    config_path = parse_cmd_args()
    validate_config(config_path)
    config = read_yaml(config_path)
    print("The input config:")
    pprint(config, sort_dicts=False, indent=2)

    img_paths = [Path(p) for p in config["Input"]["InputImagePaths"]]
    target_shape = get_target_shape(img_paths)

    feature_reg = "FeatureReg"
    opt_flow_reg = "OptFlowReg"

    if feature_reg in config["RegistrationParameters"]:
        img_paths = run_feature_reg(config, target_shape)

    if opt_flow_reg in config["RegistrationParameters"]:
        run_opt_flow_reg(config, feature_reg, img_paths, target_shape)


if __name__ == "__main__":
    main()
