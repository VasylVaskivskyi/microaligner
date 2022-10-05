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


import re
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Optional, Union

import yaml

FloatInt = Union[float, int]


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as s:
        yaml_file = yaml.safe_load(s)
    return yaml_file


def check_field_dtype(field_name: str, dtype: Union[type, Iterable[type]], obj: dict):
    if not isinstance(dtype, Iterable):
        dtype = [dtype]
    if field_name in obj:

        if any(isinstance(obj[field_name], d) for d in dtype):
            pass
        else:
            msg = (
                f"Field {field_name} has wrong data type {type(obj[field_name])},"
                + f" expected {dtype}"
            )
            raise TypeError(msg)
    else:
        msg = f"Field {field_name} is absent"
        raise KeyError(msg)


def check_field_min_max(
    field_name: str,
    _min: Optional[FloatInt] = None,
    _max: Optional[FloatInt] = None,
    obj: dict = None,
):
    if field_name not in obj:
        msg = f"Field {field_name} is absent"
        raise KeyError(msg)
    if obj is None:
        raise ValueError("Input object is None")
    if _min is None and _max is None:
        pass
    if isinstance(obj[field_name], int) or isinstance(obj[field_name], float):
        if _min is not None and obj[field_name] < _min:
            msg = f"Field {field_name} value is smaller than minimum: {_min}"
            raise ValueError(msg)
        if _max is not None and obj[field_name] > _max:
            msg = f"Field {field_name} value is greater than maximum: {_max}"
            raise ValueError(msg)


class RegParam:
    NumberPyramidLevels: int
    NumberIterationsPerLevel: int
    TileSize: int
    Overlap: int
    NumberOfWorkers: int
    UseFullResImage: bool
    UseDOG: bool

    def check_fields(self, d: dict):
        check_field_dtype("NumberPyramidLevels", int, d)
        check_field_dtype("NumberIterationsPerLevel", int, d)
        check_field_dtype("TileSize", int, d)
        check_field_dtype("Overlap", int, d)
        check_field_dtype("NumberOfWorkers", int, d)
        check_field_dtype("UseFullResImage", bool, d)
        check_field_dtype("UseDOG", bool, d)

        check_field_min_max("NumberPyramidLevels", 0, 8, d)
        check_field_min_max("NumberIterationsPerLevel", 1, None, d)
        check_field_min_max("TileSize", 20, None, d)
        check_field_min_max("Overlap", 10, d["TileSize"], d)
        check_field_min_max("NumberOfWorkers", 0, None, d)

    def read_from_dict(self, d: dict):
        self.check_fields(d)
        self.NumberPyramidLevels = d["NumberPyramidLevels"]
        self.NumberIterationsPerLevel = d["NumberIterationsPerLevel"]
        self.TileSize = d["TileSize"]
        self.Overlap = d["Overlap"]
        self.NumberOfWorkers = d["NumberOfWorkers"]
        self.UseFullResImage = d["UseFullResImage"]
        self.UseDOG = d["UseDOG"]

    def __repr__(self):
        return str(self.__dict__)


class PipelineInput:
    InputImagePaths: dict
    ReferenceCycle: int
    ReferenceChannel: str
    PipelineInputType: str

    def __repr__(self):
        return str(self.__dict__)


class PipelineOutput:
    OutputDir: Path
    OutputPrefix: str
    SaveOutputToCycleStack: bool

    def __repr__(self):
        return str(self.__dict__)


class PipelineRegParam:
    FeatureReg: RegParam = RegParam()
    OptFlowReg: RegParam = RegParam()

    def __repr__(self):
        return f"FeatureReg: {self.FeatureReg}, OptFlowReg: {self.OptFlowReg}"


class PipelineConfig:
    Input: PipelineInput = PipelineInput()
    Output: PipelineOutput = PipelineOutput()
    RegistrationParameters: PipelineRegParam = PipelineRegParam()

    def __repr__(self):
        return str(self.__dict__)


class PipelineConfigReader:
    Input: PipelineInput = PipelineInput()
    Output: PipelineOutput = PipelineOutput()
    RegistrationParameters: PipelineRegParam = PipelineRegParam()

    def read_config(self, config_path: Path):
        config = read_yaml(config_path)
        self.check_top_lvl_fields_exist(config)
        self.parse_input(config["Input"])
        self.parse_output(config["Output"])
        self.parse_reg_param(config["RegistrationParameters"])
        pipeline_config = PipelineConfig()
        pipeline_config.Input = self.Input
        pipeline_config.Output = self.Output
        pipeline_config.RegistrationParameters = self.RegistrationParameters
        return pipeline_config

    def check_top_lvl_fields_exist(self, config: dict):
        top_lvl_fields = ["Input", "Output", "RegistrationParameters"]
        missing_f = []
        for f in top_lvl_fields:
            if f not in config:
                missing_f.append(f)
        if missing_f:
            msg = (
                "Incorrectly formatted config file."
                "These fields are absent: " + str(missing_f)
            )
            raise ValueError(msg)

    def parse_input(self, input_dict: dict):
        if not isinstance(input_dict, dict):
            raise ValueError("Input field is incorrect")

        check_field_dtype("InputImagePaths", (dict, list), input_dict)
        check_field_dtype("ReferenceCycle", int, input_dict)
        check_field_dtype("ReferenceChannel", str, input_dict)

        check_field_min_max("ReferenceCycle", 1, None, input_dict)
        path_dict = input_dict["InputImagePaths"]

        path_dict_type = self.get_path_dict_type(path_dict)
        input_paths = self.parse_path_dict(path_dict, path_dict_type)
        ref_cycle = input_dict["ReferenceCycle"]
        ref_ch = input_dict["ReferenceChannel"]

        self.Input.InputImagePaths = input_paths
        self.Input.ReferenceCycle = ref_cycle
        self.Input.ReferenceChannel = ref_ch
        self.Input.PipelineInputType = path_dict_type

    def parse_path_dict(self, path_dict, path_dict_type: str):
        proc_path_dict = dict()
        cyc_name_pat = re.compile(r"Cycle \d+")

        if path_dict_type == "CycleBuilder":
            for cyc_name in path_dict:
                ch_list = []
                if cyc_name_pat.match(cyc_name):
                    cyc_id = int(re.search(r"(\d+)", cyc_name).groups()[0])
                    for ch, path_str in path_dict[cyc_name].items():
                        if cyc_id in proc_path_dict:
                            proc_path_dict[cyc_id][ch] = Path(path_str)
                        else:
                            proc_path_dict[cyc_id] = {ch: Path(path_str)}
                        ch_list.append(ch)
                else:
                    msg = "Cycle names in config file should follow pattern Cycle N"
                    raise ValueError(msg)
                if len(ch_list) > len(set(ch_list)):
                    msg = f"Channel names are repeated in the Cycle {cyc_id}: {ch_list}"
                    raise ValueError(msg)

        elif path_dict_type == "CycleStack":
            proc_path_dict[0] = Path(path_dict["CycleStack"])
        else:
            for cyc_name in path_dict:
                if cyc_name_pat.match(cyc_name):
                    cyc_id = int(re.search(r"(\d+)", cyc_name).groups()[0])
                    proc_path_dict[cyc_id] = Path(path_dict[cyc_name])
                else:
                    msg = "Cycle names in config file should follow pattern Cycle N"
                    raise ValueError(msg)
        return proc_path_dict

    def parse_output(self, output_dict: dict):
        check_field_dtype("OutputDir", str, output_dict)
        check_field_dtype("OutputPrefix", str, output_dict)
        check_field_dtype("SaveOutputToCycleStack", bool, output_dict)

        self.Output.OutputDir = Path(output_dict["OutputDir"])
        self.Output.OutputPrefix = output_dict["OutputPrefix"]
        self.Output.SaveOutputToCycleStack = output_dict["SaveOutputToCycleStack"]

    def parse_reg_param(self, reg_dict: dict):
        if "FeatureReg" not in reg_dict and "OptFlowReg" not in reg_dict:
            msg = (
                "Parameters for hte registration methods are absent. "
                + "At least one of the registration methods: "
                + "FeatureReg or OptFlowReg must be present."
            )
            raise ValueError(msg)
        if "FeatureReg" in reg_dict:
            check_field_dtype("FeatureReg", dict, reg_dict)
            self.RegistrationParameters.FeatureReg.read_from_dict(
                reg_dict["FeatureReg"]
            )
        else:
            self.RegistrationParameters.FeatureReg = None

        if "OptFlowReg" in reg_dict:
            check_field_dtype("OptFlowReg", dict, reg_dict)
            self.RegistrationParameters.OptFlowReg.read_from_dict(
                reg_dict["OptFlowReg"]
            )
        else:
            self.RegistrationParameters.OptFlowReg = None

    def get_path_dict_type(self, path_dict: dict) -> str:
        path_dict_type = None
        if "CycleStack" in path_dict:
            if len(path_dict) > 1:
                msg = "When input is CycleStack you can specify at most 1 image path"
                raise ValueError(msg)
            else:
                path_dict_type = "CycleStack"
        else:
            vals = path_dict.values()
            num_dict_inst = 0
            num_str_inst = 0
            for val in vals:
                if isinstance(val, dict):
                    num_dict_inst += 1
                elif isinstance(val, str):
                    num_str_inst += 1
            if num_dict_inst > 0 and num_str_inst > 0:
                msg = "Mixed input is not yet supported"
                raise NotImplemented(msg)
            elif num_dict_inst == 0 and num_str_inst == 0:
                msg = (
                    "Cannot recognize type of InputImagePaths."
                    + "Please check your config file against the reference."
                )
                raise ValueError(msg)
            elif num_dict_inst < 2 and num_str_inst < 2:
                msg = (
                    "Not enough cycles for registration. "
                    + "Please provide at least two cycles"
                )
                raise ValueError(msg)
            else:
                if num_dict_inst > 0:
                    path_dict_type = "CycleBuilder"
                elif num_str_inst > 0:
                    path_dict_type = "CyclePerImage"
        return path_dict_type
