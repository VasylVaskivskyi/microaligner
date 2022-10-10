## MicroAligner: image registration for large scale microscopy

- Automatic, no need to manually select points of interest
- Fast, most of internal tasks are parallelized
- Memory efficient, keeps only one image page in the memory at a time
- Scalable, the more cores you have the faster it works
- Linear and non-linear alignment, gives best results together in that order

There are two main methods that can do alignment, and also a pipeline that can run both of them based 
on parameters provided in a config file.

### Methods:

Affine feature based registration method first finds image features using `FAST` feature finder. It detects areas with 
large intensity changes. After that a `DAISY` feature descriptor collects histograms of oriented gradients
for each found feature. Then the described features are matched using `FLANN` based `knn` matcher that finds 
correspondence between features of reference and moving images. The matches are filtered according to the distance between the neighbours.
Finally, the coordinates of matched features are aligned using `RANSAC` algorithm and the affine transformation is computed.
This method is good for aligning large linear drifts across images. 

Non-linear optical flow based registration uses `Farneback` method that looks for pixels with the highest similarity in the given window.
Then for each pixel it computes a 2D vector that describes movement of the pixel from one image to the other. 
This method is good for aligning of small local shifts across images.

If you have uneven illumination across the image, and it affects the registration quality, 
you can try to tackle that by enabling image preprocessing with `difference of Gaussians` (`DOG`). 
It will extract information about strong gradients near the edges
and discard gradual changes caused by uneven illumination.

### Pipeline:

If images have z-planes, the script perform maximum intensity projection along the z dimension
before doing the registration, so the alignment happens only in X-Y coordinates.
The output image has OME-TIFF metadata, and dimension order TCZYX.

### Installation

`pip install microaligner`

### Dependencies
`numpy tifffile pandas opencv-contrib-python dask scikit-learn scikit-image` \
Also check up the `environment.yaml` file.

### Example usage

#### As a pipeline
**`microaligner config.yaml`**

For details about the config parameters please refer to the example `config.yaml` provided in this repository.

#### As a module

##### Feature based registration
```python
from microaligner import FeatureRegistrator, transform_img_with_tmat
freg = FeatureRegistrator()
freg.ref_img = img1
freg.mov_img = img2
transformation_matrix = freg.register()

img2_feature_reg_aligned = transform_img_with_tmat(img2, img2.shape, transformation_matrix)
```

##### Optical flow based registration
```python
from microaligner import OptFlowRegistrator, Warper 
ofreg = OptFlowRegistrator()
ofreg.ref_img = img1
ofreg.mov_img = img2
flow_map = ofreg.register()

warper = Warper()
warper.image = img2
warper.flow = flow_map
img2_optflow_reg_aligned = warper.warp()
```

### Acknowledgments

This package could have never been developed without thorough battle-testing by 
[Tong Li](https://github.com/BioinfoTongLI) 
and [Jun Sung Park](https://github.com/jpark27), 
and guidance from [Omer Bayraktar](https://github.com/oabayraktar).
