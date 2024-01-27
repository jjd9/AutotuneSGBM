"""

Python script to optimize the OpenCV SGBM/BM algorithm using Reference-based and Reference-free methods.

Author: jjd9

"""

import click
import cv2
import glob
import optuna
import os
from tqdm import tqdm

from optimization import objective, SaveBestTrialCallback
from utils import parse_calibration_file, save_results, load_stereo_params


@click.command()
@click.option("--dataset_name", type=str, help="Name of the dataset")
@click.option(
    "--method",
    type=click.Choice(["ref", "no_ref"]),
    help='Method to use ("ref" or "no_ref")',
)
@click.option("--stereo_algorithm", type=click.Choice(["StereoBM", "StereoSGBM"]), default="StereoSGBM", help="Stereo algorithm to tune")
@click.option("--max_iter", type=int, default=1000, help="Maximum number of iterations")
@click.option("--patience", type=int, default=1000, help="Patience value")
@click.option("--initial_params_path", type=str, default="params_XXX.yaml", help="Path to yaml file with initial guess for stereo params")
def autotune(dataset_name, method, max_iter, patience, stereo_algorithm, initial_params_path):
    image_dir = os.path.join("dataset", dataset_name)
    if not os.path.exists(image_dir):
        raise ValueError(f"Dataset directory not found at: {image_dir}")

    output_dir = os.path.join("output", dataset_name)
    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Read calibration")
    calib_path = os.path.join(image_dir, "calib.yaml")
    if os.path.exists(calib_path):
        (
            height,
            width,
            K_left,
            dist_left,
            K_right,
            dist_right,
            R,
            T,
        ) = parse_calibration_file(calib_path)
        rectify = True
        print("Found calibration file, will rectify images!")
    else:
        rectify = False
        print(
            f"Could not find calibration file at {calib_path}, will NOT rectify images!"
        )

    print("Read L/R images")

    left_image_files = glob.glob(os.path.join(image_dir, "left", "*.png"))
    right_image_files = [x.replace("left", "right") for x in left_image_files]
    reference_disparity_files = [
        x.replace("left", "reference").replace(".png", ".tiff")
        for x in left_image_files
    ]

    num_images = len(left_image_files)

    left_images = []
    right_images = []
    ref_disparities = []

    if rectify:
        # Rectify the images
        R_left, R_right, P_left, P_right, _, _, _ = cv2.stereoRectify(
            K_left, dist_left, K_right, dist_right, (width, height), R, T
        )
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            K_right, dist_right, R_right, P_right, (width, height), cv2.CV_16SC2
        )
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            K_left, dist_left, R_left, P_left, (width, height), cv2.CV_16SC2
        )

    for i in tqdm(range(num_images)):
        imgL = cv2.imread(left_image_files[i], cv2.IMREAD_UNCHANGED)
        imgR = cv2.imread(right_image_files[i], cv2.IMREAD_UNCHANGED)
        if rectify:
            imgL = cv2.remap(imgL, left_map_x, left_map_y, cv2.INTER_LANCZOS4)
            imgR = cv2.remap(imgR, right_map_x, right_map_y, cv2.INTER_LANCZOS4)

        left_images.append(imgL)
        right_images.append(imgR)
        ref_disparities.append(
            cv2.imread(reference_disparity_files[i], cv2.IMREAD_UNCHANGED)
        )
    print(f"Read {num_images} images")

    if any([x is None for x in ref_disparities]) and method == "ref":
        raise ValueError(
            "User selected reference-based method but not enough references were provided!"
        )
    elif method == "ref":
        ref_disparities = [x.astype(float) for x in ref_disparities]

    # package the data
    image_data = list(zip(left_images, right_images, ref_disparities))

    obj_func = lambda trial: objective(trial, method, image_data, stereo_algorithm)

    # run hyper-parameter optimization
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0)
    )

    if os.path.exists(initial_params_path):
        print(f"Loading initial guess from file: {initial_params_path}")
        initial_params = load_stereo_params(initial_params_path)
        study.enqueue_trial(initial_params)

    try:
        study.optimize(
            obj_func,
            n_trials=max_iter,
            callbacks=[
                SaveBestTrialCallback(
                    image_data=image_data, patience=patience, output_dir=output_dir
                )
            ],
        )
    finally:
        print("Final objective: ", obj_func(study.best_params))
        print("Final params: ", study.best_params)
        save_results(image_data, study.best_params, output_dir)


if __name__ == "__main__":
    autotune()
