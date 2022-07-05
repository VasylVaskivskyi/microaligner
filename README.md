## LIA: Large image aligner for microscopy images

It efficiently performs alignment (registration) of gigapixel size images.
It is scalable - the more cores you have the faster it works.
There are two main classes that can do alignment, and also a pipeline that can run both of them based 
on parameters provided in a config file.

Images with z-planes are projected along the z-plane before performing registration.

### Installation

`pip install lia`

### Dependencies
`numpy tifffile pandas opencv-contrib-python dask scikit-learn scikit-image` \
Also check up the `environment.yaml` file.

### Example usage

#### As a module

##### Feature based registration
```python
from lia import FeatureRegistrator, transform_img_with_tmat
freg = FeatureRegistrator()
freg.ref_img = img1
freg.mov_img = img2
transformation_matrix = freg.register()

img2_feature_reg_aligned = transform_img_with_tmat(img2, img2.shape, transformation_matrix)
```

##### Optical flow based registration
```python
from lia import OptFlowRegistrator, Warper 
ofreg = OptFlowRegistrator()
ofreg.ref_img = img1
ofreg.mov_img = img2
flow_map = ofreg.register()

warper = Warper()
warper.image = img2
warper.flow = flow_map
img2_optflow_reg_aligned = warper.warp()
```


#### As a pipeline
**`lia config.yaml`**

For details about the config parameters please refer to the example `config.yaml` provided in this repository.

### Acknowledgments

This package could never have been developed without thorough battle-testing by 
[Tong Li](https://github.com/BioinfoTongLI) 
and [Jun Sung Park](https://github.com/jpark27), 
and guidance from [Omer Bayraktar](https://github.com/oabayraktar).
