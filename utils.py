"""

Utilities for processing images and parameter files.

Author: jjd9

"""

from copy import deepcopy
import cv2
import numpy as np
import os
import yaml


def compute_x_gradient(img):
    """
    Compute the x gradient magnitude, normalized btw 0.1 and 1.0
    """
    # Read the image
    if len(img.shape) == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Compute the gradient along the x-axis using the Sobel operator
    gradient_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=9)
    gradient_x = np.abs(gradient_x)

    # Normalize the gradient values to be between 0.1 and 1
    gradient_normalized = cv2.normalize(gradient_x, None, 0.1, 1.0, cv2.NORM_MINMAX)

    return gradient_normalized


def project_left_to_right(disparity_map, left_image):
    """
    Project pixels from the left camera onto the right camera using a disparity map.

    Args:
    - disparity_map (numpy array): Disparity map obtained from StereoSGBM.
    - left_pixels (numpy array): Left image

    Returns:
    - right_pixels (numpy array): Right image reconstructed from the input left image
    """

    # Extract height and width from the disparity map
    height, width = disparity_map.shape[:2]

    left_cols, left_rows = np.meshgrid(np.arange(width), np.arange(height))

    # Initialize array for storing corresponding points in the right image
    right_pixels = np.zeros_like(left_image)
    valid_pixel_mask = np.zeros((height, width), dtype=bool)
    right_cols = left_cols - disparity_map
    valid_disparity = (disparity_map > disparity_map.min())
    cond = np.logical_and(np.logical_and(right_cols >= 0, right_cols < width), valid_disparity)

    if len(right_pixels.shape) == 3:
        right_pixels[left_rows[cond], right_cols[cond], :] = left_image[
            left_rows[cond], left_cols[cond], :
        ]
    else:
        right_pixels[left_rows[cond], right_cols[cond]] = left_image[
            left_rows[cond], left_cols[cond]
        ]
    valid_pixel_mask[left_rows[cond], right_cols[cond]] = True

    return right_pixels, valid_pixel_mask


def parse_calibration_file(file_path):
    with open(file_path, "r") as file:
        calibration_data = yaml.safe_load(file)
    height = calibration_data["height"]
    width = calibration_data["width"]

    left_intrinsics = calibration_data["left"]
    right_intrinsics = calibration_data["right"]
    stereo_extrinsics = calibration_data["stereo"]

    left_K = np.array(
        [
            [left_intrinsics["fx"], 0, left_intrinsics["cx"]],
            [0, left_intrinsics["fy"], left_intrinsics["cy"]],
            [0, 0, 1],
        ]
    )

    left_dist = np.array(
        [
            left_intrinsics["k1"],
            left_intrinsics["k2"],
            left_intrinsics["p1"],
            left_intrinsics["p2"],
            left_intrinsics["k3"],
        ]
    )

    right_K = np.array(
        [
            [right_intrinsics["fx"], 0, right_intrinsics["cx"]],
            [0, right_intrinsics["fy"], right_intrinsics["cy"]],
            [0, 0, 1],
        ]
    )

    right_dist = np.array(
        [
            right_intrinsics["k1"],
            right_intrinsics["k2"],
            right_intrinsics["p1"],
            right_intrinsics["p2"],
            right_intrinsics["k3"],
        ]
    )

    R = cv2.Rodrigues(
        np.array(
            [stereo_extrinsics["RX"], stereo_extrinsics["RY"], stereo_extrinsics["RZ"]]
        )
    )[0]

    t = np.array(
        [
            [stereo_extrinsics["Baseline"]],
            [stereo_extrinsics["TY"]],
            [stereo_extrinsics["TZ"]],
        ]
    )

    return height, width, left_K, left_dist, right_K, right_dist, R, t


def dict_to_stereo_proc(trial):
    algo = trial["algo"]
    prefilter_size = trial["prefilter_size"]
    prefilter_cap = trial["prefilter_cap"]
    correlation_window_size = trial["correlation_window_size"]
    min_disparity = trial["min_disparity"]
    disparity_range = trial["disparity_range"]
    texture_threshold = trial["texture_threshold"]
    P1 = trial["P1"]
    P2 = trial["P2"]
    disp12MaxDiff = trial["disp12MaxDiff"]
    uniqueness_ratio = trial["uniqueness_ratio"]
    speckle_window_size = trial["speckle_window_size"]
    speckle_range = trial["speckle_range"]
    prefilter_type = trial["prefilter_type"]
    # Create the StereoSGBM object
    if algo == "StereoBM":
        stereo_proc = cv2.StereoBM_create(
            numDisparities=disparity_range,  # Adjust this value based on your requirements
            blockSize=correlation_window_size,
        )  # Adjust this value based on your requirements
        stereo_proc.setPreFilterType(prefilter_type)
        stereo_proc.setPreFilterSize(prefilter_size)
        stereo_proc.setPreFilterCap(prefilter_cap)
        stereo_proc.setMinDisparity(min_disparity)
        stereo_proc.setTextureThreshold(texture_threshold)
        stereo_proc.setUniquenessRatio(uniqueness_ratio)
        stereo_proc.setSpeckleWindowSize(speckle_window_size)
        stereo_proc.setSpeckleRange(speckle_range)
    elif algo == "StereoSGBM":
        stereo_proc = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=disparity_range,
            blockSize=correlation_window_size,
            P1=P1,
            P2=P2,
            disp12MaxDiff=disp12MaxDiff,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            preFilterCap=prefilter_cap,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
    else:
        raise ValueError(f"Unknown algorithm {algo}!")
    return stereo_proc


def load_results(output_dir):
    with open(os.path.join(output_dir, "stereo_params.yaml"), "r") as file:
        stereo_proc_params = yaml.safe_load(file)
    return dict_to_stereo_proc(stereo_proc_params)


def save_results(image_data, params, output_dir):
    # save params to yaml file
    with open(os.path.join(output_dir, "stereo_params.yaml"), "w") as file:
        yaml.safe_dump(params, file)

    # save debug images
    stereo_proc_left = dict_to_stereo_proc(params)

    for img_id, (_left_image, _right_image, ref_disparity) in enumerate(image_data):
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

        fake_right_image, valid_right_proj_pixels = project_left_to_right(
            disparity_map.astype(int), left_image
        )

        if len(right_image.shape) == 3:
            rg = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            frg = cv2.cvtColor(fake_right_image, cv2.COLOR_BGR2GRAY)
        else:
            rg = right_image
            frg = fake_right_image

        right_error = cv2.absdiff(rg, frg)
        right_error[~valid_right_proj_pixels] = 0

        disparity_visual = (disparity_map - disparity_map.min()) / (
            disparity_map.max() - disparity_map.min()
        )
        # Convert to a color image (pseudo-color)
        disparity_visual = (disparity_visual * 255).astype(np.uint8)
        disparity_visual = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_JET)
        disparity_visual[disparity_map == disparity_map.min()] *= 0

        if ref_disparity is not None:
            ref_disparity_visual = (ref_disparity - ref_disparity.min()) / (
                ref_disparity.max() - ref_disparity.min()
            )
            # Convert to a color image (pseudo-color)
            ref_disparity_visual = (ref_disparity_visual * 255).astype(np.uint8)
            ref_disparity_visual = cv2.applyColorMap(
                ref_disparity_visual, cv2.COLORMAP_JET
            )
            cv2.imwrite(
                os.path.join(output_dir, f"disparity_{img_id}.png"),
                cv2.vconcat([disparity_visual, ref_disparity_visual]),
            )
        else:
            cv2.imwrite(
                os.path.join(output_dir, f"disparity_{img_id}.png"), disparity_visual
            )
        cv2.imwrite(
            os.path.join(output_dir, f"right_error_{img_id}.png"),
            cv2.applyColorMap(right_error, cv2.COLORMAP_JET),
        )
        cv2.imwrite(
            os.path.join(output_dir, f"fake_right_{img_id}.png"),
            cv2.vconcat([fake_right_image, right_image]),
        )
