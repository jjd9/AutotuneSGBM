"""

Helper functions for the optimization process.

Author: jjd9

"""

from copy import deepcopy
import cv2
import numpy as np
import optuna
from skimage.metrics import structural_similarity as ssim

from utils import (
    compute_x_gradient,
    dict_to_stereo_proc,
    project_pixels_by_disparity,
    save_results,
)

def bm_params(trial,n_channels):
        params = dict()
        # image preprocessing
        ############################
        params["prefilter_size"] = trial.suggest_int(
            "prefilter_size", 5, 11, step=2
        )  # must be odd
        params["prefilter_cap"] = trial.suggest_int("prefilter_cap", 1, 32)
        params["prefilter_type"] = trial.suggest_categorical(
            "prefilter_type",
            [
                cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE,
                cv2.STEREO_BM_PREFILTER_XSOBEL,
            ],
        )
        ############################

        # actual stereo matching
        ############################
        params["algo"] = trial.suggest_categorical("algo", ["StereoBM"])
        params["correlation_window_size"] = trial.suggest_int(
            "correlation_window_size", 5, 25, step=2
        )
        params["min_disparity"] = trial.suggest_int("min_disparity", -32, 64, step=16)
        params["disparity_range"] = trial.suggest_int(
            "disparity_range", 16, 256, step=16
        )
        ############################

        # disparity post processing
        ############################
        params["P1"] = trial.suggest_int("P1", 0, 0)
        params["P2"] = trial.suggest_int("P2", 0, 0)

        params["texture_threshold"] = trial.suggest_int("texture_threshold", 0, 100)
        params["disp12MaxDiff"] = trial.suggest_int(
            "disp12MaxDiff", -1, 128
        )  # -1 == disabled
        params["uniqueness_ratio"] = trial.suggest_int("uniqueness_ratio", 0, 30)
        params["speckle_window_size"] = trial.suggest_int("speckle_window_size", 3, 25)
        params["speckle_range"] = trial.suggest_int("speckle_range", 0, 30)
        ############################
        return params


def sgbm_params(trial,n_channels):
        params = dict()
        # image preprocessing
        ############################
        params["prefilter_size"] = trial.suggest_int("prefilter_size", 1, 1) #trial.suggest_int(
        #     "prefilter_size", 5, 11, step=2
        # )  # must be odd
        params["prefilter_cap"] = trial.suggest_int("prefilter_cap", 1, 32)
        params["prefilter_type"] = trial.suggest_int("prefilter_type", 1, 1) #trial.suggest_categorical(
        #     "prefilter_type",
        #     [
        #         cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE,
        #         cv2.STEREO_BM_PREFILTER_XSOBEL,
        #     ],
        # )
        ############################

        # actual stereo matching
        ############################
        params["algo"] = trial.suggest_categorical("algo", ["StereoSGBM"])
        # params["algo"] = trial.suggest_categorical("algo", ["StereoBM", "StereoSGBM"])
        params["correlation_window_size"] = trial.suggest_int(
            "correlation_window_size", 5, 25, step=2
        )
        params["min_disparity"] = trial.suggest_int("min_disparity", -32, 64, step=16)
        params["disparity_range"] = trial.suggest_int(
            "disparity_range", 16, 256, step=16
        )
        ############################

        # disparity post processing
        ############################

        params["texture_threshold"] = trial.suggest_int("texture_threshold", 0,0)# trial.suggest_int("texture_threshold", 0, 100)
        # P1 and P2 have a large range, so we opt to reduce the search space significantly with a good initial guess
        P1_guess = min(
            4000, int(8 * n_channels * params["correlation_window_size"] ** 2)
        )
        P2_guess = min(
            4000, int(32 * n_channels * params["correlation_window_size"] ** 2)
        )
        params["P1"] = trial.suggest_int("P1", max(0, P1_guess - 10), P1_guess + 10)
        params["P2"] = trial.suggest_int("P2", max(0, P2_guess - 10), P2_guess + 10)
        params["disp12MaxDiff"] = trial.suggest_int(
            "disp12MaxDiff", -1, 128
        )  # -1 == disabled
        params["uniqueness_ratio"] = trial.suggest_int("uniqueness_ratio", 0, 30)
        params["speckle_window_size"] = trial.suggest_int("speckle_window_size", 3, 25)
        params["speckle_range"] = trial.suggest_int("speckle_range", 0, 30)
        ############################
        return params

def objective(trial, method, image_data, stereo_algorithm="StereoSGBM"):
    if len(image_data[0][0].shape) == 3:
        n_channels = 3
    else:
        n_channels = 1

    if isinstance(trial, dict):
        params = trial
    else:
        # TODO: Config file to modify search space dimensions
        if stereo_algorithm == "StereoSGBM":
            params = sgbm_params(trial, n_channels)
        elif stereo_algorithm == "StereoBM":
            params = bm_params(trial, n_channels)
        else:
            raise ValueError(f"Unknown stereo algorithm {stereo_algorithm}")

    # Create the StereoSGBM object
    stereo_proc_left = dict_to_stereo_proc(params)
    right_params = deepcopy(params)
    right_params["min_disparity"] = (
        -(params["min_disparity"] + params["disparity_range"]) + 1
    )
    stereo_proc_right = dict_to_stereo_proc(right_params)

    # Compute the disparity map
    error = 0.0
    for imd_id, (_left_image, _right_image, ref_disparity) in enumerate(image_data):
        if params["algo"] == "StereoBM" and len(_left_image.shape) == 3:
            # stereoBM only handles grayscale images
            left_image = cv2.cvtColor(_left_image, cv2.COLOR_BGR2GRAY)
            right_image = cv2.cvtColor(_right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_image = _left_image.copy()
            right_image = _right_image.copy()

        left_disparity = (
            stereo_proc_left.compute(left_image, right_image).astype(float) / 16.0
        )

        if method == "ref":
            # Reference-based weighted-MAE
            # weight to force the optimization to pay attention to high texture regions
            left_grad_weight = compute_x_gradient(left_image)
            error += np.mean(np.abs(left_disparity - ref_disparity) * left_grad_weight)
            continue

        right_disparity = (
            stereo_proc_right.compute(right_image, left_image).astype(float) / 16.0
        )

        # if method == "no_ref"
        for pyr_lvl in range(2): # another idea taken from monodepth
            if pyr_lvl != 0:
                right_image = cv2.pyrDown(
                    right_image,
                    dstsize=(left_image.shape[1] // 2, left_image.shape[0] // 2),
                )
                left_disparity = cv2.pyrDown(
                    left_disparity,
                    dstsize=(left_image.shape[1] // 2, left_image.shape[0] // 2),
                )
                right_disparity = cv2.pyrDown(
                    right_disparity,
                    dstsize=(left_image.shape[1] // 2, left_image.shape[0] // 2),
                )
                left_image = cv2.pyrDown(
                    left_image,
                    dstsize=(left_image.shape[1] // 2, left_image.shape[0] // 2),
                )

            # Reference-free L/R consistency check

            # projection left image to right and vice versa
            fake_right_image, _ = project_pixels_by_disparity(
                left_disparity.astype(int), left_image
            )
            fake_left_image, _ = project_pixels_by_disparity(
                right_disparity.astype(int), right_image
            )

            # projection left disparity to right and vice versa
            fake_right_disparity, _ = project_pixels_by_disparity(
                left_disparity.astype(int), -left_disparity
            )
            fake_right_disparity[
                right_disparity == right_disparity.min()
            ] = right_disparity.min()

            fake_left_disparity, _ = project_pixels_by_disparity(
                right_disparity.astype(int), -right_disparity
            )
            fake_left_disparity[
                left_disparity == left_disparity.min()
            ] = left_disparity.min()

            if len(right_image.shape) == 3:
                rg = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
                frg = cv2.cvtColor(fake_right_image, cv2.COLOR_BGR2GRAY)
                lt = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
                flt = cv2.cvtColor(fake_left_image, cv2.COLOR_BGR2GRAY)
            else:
                rg = right_image
                frg = fake_right_image
                lt = left_image
                flt = fake_left_image

            # Compute reconstruction error terms

            # weight to force the optimization to pay attention to high texture regions
            right_grad_weight = compute_x_gradient(rg)
            left_grad_weight = compute_x_gradient(lt)

            _, right_recon_similarity_image = ssim(rg, frg, full=True, data_range=255)
            # normalize ssim (-1,1) between 0 and 1
            right_recon_similarity_image = (right_recon_similarity_image + 1.0) / 2.0
            inv_right_recon_similarity_image = 1.0 - right_recon_similarity_image
            inv_right_recon_similarity = np.mean(
                inv_right_recon_similarity_image * right_grad_weight
            )

            _, left_recon_similarity_image = ssim(lt, flt, full=True, data_range=255)
            # normalize ssim (-1,1) between 0 and 1
            left_recon_similarity_image = (left_recon_similarity_image + 1.0) / 2.0
            inv_left_recon_similarity_image = 1.0 - left_recon_similarity_image
            inv_left_recon_similarity = np.mean(
                inv_left_recon_similarity_image * left_grad_weight
            )

            # regularize the optimization by
            # computing similarity between disparity map and aligned image (monodepth uses weighted disparity smoothness instead,
            # but the idea is very similar imo)

            # normalize the disparity map so we can compare it with the left image
            normalized_left_disparity = cv2.normalize(
                left_disparity, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            normalized_right_disparity = cv2.normalize(
                right_disparity, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # normalize left ssim to be (0, 1) and flip
            inv_left_similarity = (
                1.0 - (ssim(lt, normalized_left_disparity, data_range=255) + 1.0) / 2.0
            )  
            inv_right_similarity = (
                1.0 - (ssim(rg, normalized_right_disparity, data_range=255) + 1.0) / 2.0
            )

            # compute fake disparities for L/R consistency checks
            lr_disparity_inconsistency = (
                np.mean(
                    np.abs(fake_right_disparity - right_disparity) * right_grad_weight
                )
                / 320.0
            )

            rl_disparity_inconsistency = (
                np.mean(np.abs(fake_left_disparity - left_disparity) * left_grad_weight)
                / 320.0
            )

            # Sum up all the error terms
            left_error = (
                0.5 * inv_right_recon_similarity
                + 0.3 * lr_disparity_inconsistency
                + 0.2 * inv_left_similarity
            )
            right_error = (
                0.5 * inv_left_recon_similarity
                + 0.3 * rl_disparity_inconsistency
                + 0.2 * inv_right_similarity
            )

            error += left_error + right_error

    return error


class SaveBestTrialCallback:
    def __init__(self, image_data, patience: int, output_dir):
        self.patience = patience
        self._best_score = None
        self._image_data = image_data
        self._no_improvement_trials = 0
        self.output_dir = output_dir

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if self._best_score is not None:
            # save results when objective improves
            if self._best_score > study.best_value:
                save_results(self._image_data, study.best_params, self.output_dir)
                self._no_improvement_trials = 0
                self._best_score = study.best_value
            else:
                self._no_improvement_trials += 1
                if self._no_improvement_trials > self.patience:
                    print("Stopping study due to lack of progress!")
                    study.stop()
        else:
            self._best_score = study.best_value
            self._no_improvement_trials = 0
