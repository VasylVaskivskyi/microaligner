import logging
import os
import sys
from glob import glob

import shutil
import yaml
import numpy as np
import tifffile as tif
import cv2 as cv
from skimage.exposure import match_histograms

osj = os.path.join

# this is used while calling this file as a script
sys.path += [os.path.abspath("."), os.path.abspath("..")]  # Add path to root
from birl.benchmark import ImRegBenchmark
from birl.utilities.experiments import create_basic_parser


class BmMicroaligner(ImRegBenchmark):
    REQUIRED_PARAMS = ImRegBenchmark.REQUIRED_PARAMS  # + ["path_config"]

    EXEC_MICROALIGNER = "microaligner"

    NAME_IMAGE_WARPED = "proc_optflow_*_cyc002.tif"

    NAME_LNDS_WARPED = "proc_optflow_*_cyc002.csv"

    COMMAND_REGISTRATION = f"{EXEC_MICROALIGNER} config.yaml"

    ref_out_path = ""
    mov_out_path = ""


    def _prepare(self):
        logging.info("-> copy configuration...")

        self._copy_config_to_expt("path_config")

    def _prepare_img_registration(self, item):
        """prepare the experiment folder if it is required,
        eq. copy some extra files

        :param dict item: dictionary with regist. params
        :return dict: the same or updated registration info
        """
        logging.debug(".. converting to grey tif")

        # def lv(img) -> float:
        #     return np.var(cv.Laplacian(img, cv.CV_64F, ksize=21))

        def get_most_var(img_bgr):
            var_img_bgr = [
                np.var(img_bgr[:, :, 0]),
                np.var(img_bgr[:, :, 1]),
                np.var(img_bgr[:, :, 2]),
            ]
            max_var = np.argmax(var_img_bgr)
            res_img = cv.normalize(img_bgr[:, :, max_var], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            return res_img

        def read_as_grey(img_path):
            img_bgr = cv.imread(str(img_path), cv.IMREAD_COLOR)
            img_ch = get_most_var(img_bgr)
            img_ch = img_ch.max() - img_ch
            return img_ch

        path_img_ref, path_img_mov, path_ln_ref, path_ln_mov = self._get_paths(item)

        ref_grey = read_as_grey(path_img_ref)
        mov_grey = read_as_grey(path_img_mov)

        out_dir = self._get_path_reg_dir(item)

        ref_name, _ = os.path.splitext(os.path.basename(path_img_ref))
        mov_name, _ = os.path.splitext(os.path.basename(path_img_mov))

        self.ref_out_path = osj(out_dir, ref_name + ".tif")
        self.mov_out_path = osj(out_dir, mov_name + ".tif")

        print("Doing HEQ")
        if ref_grey.var() > mov_grey.var():
            mov_grey = match_histograms(mov_grey, ref_grey)
        else:
            ref_grey = match_histograms(ref_grey, mov_grey)

        print("Saving greyscale")
        tif.imwrite(self.ref_out_path, ref_grey, photometric="minisblack")
        tif.imwrite(self.mov_out_path, mov_grey, photometric="minisblack")
        return item

    def _make_ma_config(self, path_img_ref, path_img_mov, path_ln_ref, path_ln_mov, out_dir):
        ma_config = {
            "Input": {
                "InputImagePaths": {
                    "Cycle 1": {
                        "img": path_img_ref
                    },

                    "Cycle 2": {
                        "img": path_img_mov}
                },
                "ReferenceCycle": 1,
                "ReferenceChannel": "img",
                "InputPoints": {
                    "Cycle 1": path_ln_ref,
                    "Cycle 2": path_ln_mov
                }
            },
            "Output": {
                "OutputDir": out_dir,
                "OutputPrefix": "proc_",
                "SaveOutputToCycleStack": False
            },

            "RegistrationParameters": {
                "FeatureReg": {
                    "NumberPyramidLevels": 4,
                    "NumberIterationsPerLevel": 3,
                    "TileSize": 1000,
                    "Overlap": 100,
                    "NumberOfWorkers": 32,
                    "UseFullResImage": True,
                    "UseDOG": False,
                },
                "OptFlowReg": {
                    "NumberPyramidLevels": 4,
                    "NumberIterationsPerLevel": 3,
                    "TileSize": 1000,
                    "Overlap": 100,
                    "NumberOfWorkers": 32,
                    "UseFullResImage": True,
                    "UseDOG": False,
                }
            }
        }
        return ma_config


    def _generate_regist_command(self, item):
        """generate the registration command(s)

        :param dict item: dictionary with registration params
        :return str|list(str): the execution commands
        """
        out_dir = self._get_path_reg_dir(item)
        path_img_ref, path_img_mov, path_ln_ref, path_ln_mov = self._get_paths(item)

        ma_config = self._make_ma_config(self.ref_out_path, self.mov_out_path, path_ln_ref, path_ln_mov, out_dir)
        ma_config_path = out_dir + "/config.yaml"
        with open(ma_config_path, "w") as s:
            yaml.dump(ma_config, s)

        cmd = f"microaligner {ma_config_path}"
        return cmd

    def _extract_warped_image_landmarks(self, item):
        """get registration results - warped registered images and landmarks

        :param dict item: dictionary with registration params
        :return dict: paths to warped images/landmarks
        """
        path_reg_dir = self._get_path_reg_dir(item)
        path_im_out = glob(osj(path_reg_dir, self.NAME_IMAGE_WARPED))
        path_lnds_out = glob(osj(path_reg_dir, self.NAME_LNDS_WARPED))
        if path_im_out:
            path_im_out = sorted(path_im_out)[0]
            print(path_im_out)

        print("LANDMARKS: ", path_lnds_out)
        if path_lnds_out:
            path_lnds_out = sorted(path_lnds_out)[0]
            print(path_lnds_out)

        return {
            self.COL_IMAGE_MOVE_WARP: path_im_out,
            self.COL_POINTS_MOVE_WARP: path_lnds_out,
        }


    def _clear_after_registration(self, item):
        """clean unnecessarily files after the registration

        :param dict item: dictionary with regist. information
        :return dict: the same or updated regist. info
        """
        logging.debug(".. no cleaning after registration experiment")
        return item


# RUN by given parameters
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info(__doc__)
    arg_params, path_expt = BmMicroaligner.main()

    if arg_params.get("run_comp_benchmark", False):
        logging.info("Comp benchmark started")
        from bm_experiments import bm_comp_perform
        bm_comp_perform.main(path_expt)
