## Autotuning (SG)BM with Deep Learning

***NOTE: This repo is a work in progress. I do not yet have any data to suggest this is better than hand tuning your (SG)BM algo.***

### Motivation
I am not a fan of tuning OpenCV's stereo (SG)BM algorithm. It is an amazing piece of software, but it has many parameters that I find hard to tune to best fit my needs.

So like any lazy programmer, I decided to automate this tuning process. How you ask? Well when you manually tune the block matching parameters with a GUI, you are solving an image quality optimization problem. You move the parameters around until they work well for your current view (I look for an output with few blobs and few holes that has similar structure to the left stereo image). But there are two problems with this manual approach:
1. this can be an unintuitive process, even though there are many well written guides on how to do this (and many of them give an idea of what range of parameters you might want to try).
2. you are likely overfitting to what you pointed your camera at (i.e. at best, you are optimizing for only a single camera view at a time)

I think that we should solve the image quality problem a different way. 
1. define an image quality objective(s)
2. use optuna to optimize that metric(s) over the space of reasonable BM parameters

There are two options in image quality assessment, Reference-free and Reference-based, and this library supports both.

#### Reference-free.

This is nice since it does not require us to find references BUT is more complex because we need to come up with some measure of image quality to optimize for in the absence of a reference. 

In my implementation, the "reference free" naming is a bit of a misnomer. Instead of having a disparity reference, I use the left and right images as the references. This leverages two simple ideas:
1. you can reconstruct the right image from the left image by shifting the left image's pixels by the disparity map values (Except at occlusions. it would be awesome to have a good way to detect occlusions and exclude them from this check). So any discrepancy between the right image and the reconstructed ("fake") right image suggests a suboptimal disparity map.
2. the structure of the disparity map should match the structure of the left image.

![Disparity Estimate](https://github.com/jjd9/AutotuneSGBM/blob/main/output/ZED1/disparity_0.png)

![Left-Right Reconstruction](https://github.com/jjd9/AutotuneSGBM/blob/main/output/ZED1/fake_right_0.png)

![Left-Right Consistency](https://github.com/jjd9/AutotuneSGBM/blob/main/output/ZED1/right_error_0.png)

#### Reference-based

This is attractively simple at first glance. You have N left/right images and N good disparity images for reference. You tune the parameters to minimize some measure of distance between the stereo blocking matching outputs and the good disparity references. BUT where do you get the references from?

Some options are:
- Deep learning stereo neural network (these produce very nice looking outputs, but can hallucinate)
- Extrinsically calibrated TOF sensor (i.e. align its depth image to your left camera to create a pseudo-disparity image)
- Synthetic image rendering tool like blender (this does not apply to real camera's though)

I leave this up to the user to provide a reference if they want to go the reference-based route (see the `dataset/CRE`` example). I would recommend using [CREStereo](https://github.com/megvii-research/CREStereo) based on personal testing unless you have a low resolution stereo pair (i.e. << 1080x720 per camera) OR an active stereo camera (since active stereo camera dot patterns usuall confuse stereo NN methods), in which case, the reference-free method would be my recommendation.

![Referece-based Example](https://github.com/jjd9/AutotuneSGBM/blob/main/output/CRE/disparity_0.png)

### Install Dependencies

`pip install -r requirements.txt`

### Steps

1. Prepare you images. Take as many images as you like (maybe 4 or 5?). For best results, the images should be distinctive and high-texture. I would also recommend using a set of images that is representative of your desired use case. (e.g. if you are using the camera at close range, dont tune on images taken at a distance, and vice versa. If you are using it in many scenearios, be sure to have a good mix in your dataset!)
Know your baseline (distance between left and right camera)!
If you have a large baseline, the optimization will probably not do well with images taken at very close range. 
If you have a small baseline, the optimization will probably not do well with images taken from very far away.
Images should have enough texture to get reasonable block matching results. e.g. If you take a picture of a solid-colored wall with a passive stereo camera, you probably wont get good results since that picture will not have enough texture for the block matching, regardless of how we tweak the parameters.
TODO: It would be nice to analyze the dataset for the user
1a. (Optional) Create reference images for your left stereo images as described in `Reference-based`. it should be saved in float-32 format as a .tiff file. The values should be disparity, not depth.

2. Create a folder in `dataset` and in it create a `left` and `right` folder and (optionally) a `calib.yml` file and (optionally) a `reference` folder. There are examples in the `dataset` you can pattern match against. If you do provide a calib.yml file, the images should be raw, unrectified images from your camera (and the optional reference should align with the rectified left image). If you do not provide a calib.yml file, the images are assumed to aleady be rectified. 
**THIS IS NOT A STEREO CAMERA CALIBRATION TOOL** It will not make up for a bad stereo camera calibration. Having a well calibrated camera, so we can get good quality rectified images, is critical to getting good disparity results with this library. If you find the optimization never converges to a good solution, consider checking your camera calibration before submitting an issue.

3. Place your images in these folders. Corresponding images should have the same names. e.g.
dataset/<dataset_name>/left/1.png <-- left stereo image
dataset/<dataset_name>/right/1.png <-- right stereo image
dataset/<dataset_name>/reference/1.png <-- disparity image

4. run autotune.py.
Usage:
`python autotune.py --dataset_name <dataset_name> --method <'ref' or 'no_ref'> --max_iter 1000 --patience 100`
Your results will be stored to `./output/<dataset_name>`
While it runs, go get a coffee, touch grass, etc... (-:
The process writes the current results to the output directory/<dataset_name> whenever the optimizer is able to improve the result (so if you get impatient and want to stop it early, you can just hit Ctrl+c).
WARNING: The code will overwrite existing results in the output/<dataset_name> directory! So be sure to copy over existing results before starting a new optimization.

5. (Optional) To verify the parameters were not overfit to your dataset, its worthwhile to evaluate your results (requires your camera to be plugged in)
assuming the camera returns the left and right image separately
`python evaluate.py --dataset_name <dataset_name> --single_image False --left_cap_id 6 --right_cap_id 7`
assuming the camera returns the left and right image concatenated
`python evaluate.py --dataset_name <dataset_name> --single_image True --left_cap_id 6 --right_cap_id -1`

### Contributing
Did you find a bug? Please create a descriptive issue and provide example inputs.

Do you have ideas on how to make this better? Or did one of my TODO's strike your interest? If so, please create an issue letting me know you would like to work on that, and start a PR. I would be happy to review and accept reasonable changes. (-:

### References
- The images in dataset/CRE are from megvii-research/CREStereo and the reference was geneated using https://github.com/ibaiGorordo/CREStereo-Pytorch
- The ranges for the search space are based on a mix of http://wiki.ros.org/stereo_image_proc/Tutorials/ChoosingGoodStereoParameters and https://learnopencv.com/depth-perception-using-stereo-camera-python-c/
