"""
Consistency-aware wrapper for the Stereo(SG)BM algorithm

"""

import numpy as np
import cv2
from utils import dict_to_stereo_proc, project_pixels_by_disparity

class ConsistentMatcher:
    def __init__(self, stereo_parms, min_consistency = 3):
        self.left_matcher = dict_to_stereo_proc(stereo_parms)
        right_stereo_parms = stereo_parms
        right_stereo_parms["min_disparity"] = - (stereo_parms["min_disparity"] + stereo_parms["disparity_range"]) + 1
        self.right_matcher = dict_to_stereo_proc(right_stereo_parms)
        self.min_consistency = min_consistency
    
    def compute(self, left_img, right_img):
        left_disparity = self.left_matcher.compute(left_img, right_img).astype(float)/16.0
        right_disparity = self.right_matcher.compute(right_img, left_img).astype(float)/16.0

        fake_left_disparity, _ = project_pixels_by_disparity(right_disparity.astype(int), -right_disparity)
        fake_left_disparity[left_disparity == left_disparity.min()] = left_disparity.min()
        disparity_error = np.abs(left_disparity - fake_left_disparity)
        lr_consistency = 255 - cv2.normalize(disparity_error, None, 0, 255, cv2.NORM_MINMAX)
        # mask off invalid disparity
        left_disparity[left_disparity==left_disparity.min()] = 0
        # mask off inconsistent disparity
        left_disparity[disparity_error > self.min_consistency] = 0

        return left_disparity, lr_consistency