"""

Helper functions for the optimization process.

Author: jjd9

"""

import cv2
import numpy as np
import optuna
from skimage.metrics import structural_similarity as ssim

from utils import (
    compute_x_gradient,
    dict_to_stereo_proc,
    project_left_to_right,
    save_results,
)


def objective(trial, method, image_data):
    if len(image_data[0][0].shape) == 3:
        n_channels = 3
    else:
        n_channels = 1

    if isinstance(trial, dict):
        params = trial
    else:
        # TODO: Config file to modify search space dimensions
        params = dict()

        # image preprocessing
        ############################
        params["prefilter_size"] = trial.suggest_int("prefilter_size", 5, 11, step=2) # must be odd
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
        params["algo"] = trial.suggest_categorical("algo", ["StereoBM", "StereoSGBM"])
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
        params["texture_threshold"] = trial.suggest_int("texture_threshold", 0, 100)
        # P1 and P2 have a large range, so we opt to reduce the search space significantly with a good initial guess
        P1_guess = min(4000, int(8 * n_channels * params["correlation_window_size"] ** 2))
        P2_guess = min(4000, int(32 * n_channels * params["correlation_window_size"] ** 2))
        params["P1"] = trial.suggest_int("P1", max(0, P1_guess - 10), P1_guess + 10)
        params["P2"] = trial.suggest_int("P2", max(0, P2_guess - 10), P2_guess + 10)
        params["disp12MaxDiff"] = trial.suggest_int(
            "disp12MaxDiff", -1, 128
        )  # -1 == disabled
        params["uniqueness_ratio"] = trial.suggest_int("uniqueness_ratio", 0, 30)
        params["speckle_window_size"] = trial.suggest_int("speckle_window_size", 0, 200)
        params["speckle_range"] = trial.suggest_int("speckle_range", 0, 15)
        ############################

    # Create the StereoSGBM object
    stereo_proc_left = dict_to_stereo_proc(params)

    # Compute the disparity map
    error = 0.0
    for _left_image, _right_image, ref_disparity in image_data:
        if params["algo"] == "StereoBM" and len(_left_image.shape) == 3:
            # stereoBM only handles grayscale images
            left_image = cv2.cvtColor(_left_image, cv2.COLOR_BGR2GRAY)
            right_image = cv2.cvtColor(_right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_image = _left_image
            right_image = _right_image

        disparity_map = (
            stereo_proc_left.compute(left_image, right_image).astype(float) / 16.0
        )

        if method == "ref":
            # Reference-based weighted-MAE
            # weight to force the optimization to pay attention to high texture regions
            left_grad_weight = compute_x_gradient(left_image)
            metric = np.mean(np.abs(disparity_map - ref_disparity) * left_grad_weight)
        elif method == "no_ref":
            # Reference-free L/R consistency check

            # projection left image to right and vice versa
            fake_right_image, right_valid_proj_mask = project_left_to_right(
                disparity_map.astype(int), left_image
            )

            if len(right_image.shape) == 3:
                rg = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
                frg = cv2.cvtColor(fake_right_image, cv2.COLOR_BGR2GRAY)
                lt = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            else:
                rg = right_image
                frg = fake_right_image
                lt = left_image

            # normalize the disparity map so we can compare it with the left image
            normalized_disparity = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # weight to force the optimization to pay attention to high texture regions
            right_grad_weight = compute_x_gradient(right_image)

            _, right_similarity_image = ssim(rg, frg, full=True, data_range=255)
            # normalize ssim (-1,1) between 0 and 1
            right_similarity_image = (right_similarity_image + 1.0) / 2.0

            # invert ssim
            inv_right_similarity_image = 1.0 - right_similarity_image

            valid_pixel_count = np.count_nonzero(right_valid_proj_mask)
            bad_pixel_ratio = 1.0 - valid_pixel_count/right_valid_proj_mask.size
            if valid_pixel_count == 0:
                return np.inf

            # Two sides of the metric:
            # 1. the reconstructed right image should look like the right image (and we care about that the most where the x 
            # gradient is high)
            # 2. the structure of the left image should roughly match the disparity map
            metric = np.mean(inv_right_similarity_image * right_grad_weight) + (1.0 - ssim(lt, normalized_disparity, data_range=255))
        else:
            raise ValueError(f"Method {method} not known!")

        error += metric

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
