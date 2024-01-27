"""

This file runs the tuned stereo BM parameters on a live stream of images from the 
stereo camera.

Author: jjd9

"""

import click
import cv2
import numpy as np
import os

from utils import parse_calibration_file, load_stereo_params
from consistent_stereo import ConsistentMatcher

@click.command()
@click.option("--dataset_name", type=str, help="Name of the dataset")
@click.option(
    "--single_image", default="True",
    type=click.Choice(["True", "False"]),
    help="Whether the camera sends a single image or separate image streams",
)
@click.option(
    "--left_cap_id",
    type=int, default=0,
    help="cv2.VideoCapture id of the left camera (or the concatenated left/right camera)",
)
@click.option(
    "--right_cap_id",
    type=int, default=-1,
    help="cv2.VideoCapture id of the left camera (or the concatenated left/right camera)",
)
def evaluate(dataset_name, single_image, left_cap_id, right_cap_id):
    single_image = (single_image == "True")
    # get calibration parameters in case we need to rectify
    calib_path = os.path.join("dataset", dataset_name, "calib.yaml")
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
        # Rectify the images
        R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(
            K_left, dist_left, K_right, dist_right, (width, height), R, T
        )
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            K_right, dist_right, R_right, P_right, (width, height), cv2.CV_16SC2
        )
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            K_left, dist_left, R_left, P_left, (width, height), cv2.CV_16SC2
        )

        rectify = True
    else:
        rectify = False
        height = None
        width = None

    # get stereo params
    stereo_proc_params_path = os.path.join("output", dataset_name)
    stereo_params = load_stereo_params(os.path.join(stereo_proc_params_path, "stereo_params.yaml"))
    algo = stereo_params["algo"]
    stereo_proc = ConsistentMatcher(stereo_params) #load_results(stereo_proc_params_path)

    # setup for getting the images
    if single_image:
        capBoth = cv2.VideoCapture(left_cap_id)  # update to match your setup
        if rectify:
            capBoth.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
            capBoth.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        capLeft = cv2.VideoCapture(left_cap_id)  # update to match your setup
        capRight = cv2.VideoCapture(right_cap_id)  # update to match your setup

    # loop as long as we keep getting images
    min_disp = 1e6
    max_disp = -1e6
    s = 0
    try:
        while True:
            if single_image:
                ret, frame = capBoth.read()
                if not ret:
                    print("Failed to read from camera!")
                    break
                h, w = frame.shape[:2]
                frameLeft = frame[:, : w // 2]
                frameRight = frame[:, w // 2 :]
            else:
                retLeft, frameLeft = capLeft.read()
                retRight, frameRight = capRight.read()
                if not retLeft or not retRight:
                    print("Failed to read from camera!")
                    break

            if rectify:
                imgLeft = cv2.remap(
                    frameLeft, left_map_x, left_map_y, cv2.INTER_LANCZOS4
                )
                imgRight = cv2.remap(
                    frameRight, right_map_x, right_map_y, cv2.INTER_LANCZOS4
                )
            else:
                imgLeft = frameLeft
                imgRight = frameRight

            if algo == "StereoBM" and len(imgLeft.shape)==3:
                imgLeft = cv2.cvtColor(imgLeft, cv2.COLOR_BGR2GRAY)
                imgRight = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)
            disparity_map, lr_consistency = stereo_proc.compute(imgLeft, imgRight)
            disparity_map = disparity_map.astype(float) / 16.0
            min_disp = min(disparity_map.min(), min_disp)
            max_disp = max(disparity_map.max(), max_disp)
            disparity_visual = (disparity_map - min_disp) / (
                max_disp - min_disp
            )
            # Convert to a color image (pseudo-color)
            disparity_visual = (disparity_visual * 255).astype(np.uint8)
            disparity_visual = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_JET)
            cv2.imshow("Rectified Left/Right", cv2.hconcat([imgLeft, imgRight]))
            cv2.imshow("disparity", disparity_visual)
            cv2.imshow("L/R consistency", cv2.applyColorMap(lr_consistency.astype(np.uint8), cv2.COLORMAP_JET))

            # stop if the user presses q
            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
            elif key & 0xFF == ord("s"):
                cv2.imwrite(f"left{s}.png", frameLeft)
                cv2.imwrite(f"right{s}.png", frameRight)
                s+=1
    finally:
        if single_image:
            capBoth.release()
        else:
            capLeft.release()
            capRight.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    evaluate()
