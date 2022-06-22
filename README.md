[![DOI](https://zenodo.org/badge/405296622.svg)](https://zenodo.org/badge/latestdoi/405296622)
## Optical flow based registration for immunofluorescence images

These scripts perform fine registration using warping. 
A map for warping is calculated using Farneback optical flow algorithm, by OpenCV.
Although images are MINMAX normalized during processing, optical flow algorithms expect images to have 
similar pixel intensities. 

Currently does not support z-stacks.

### Command line arguments

**`-i`**  path to image stack

**`-c`**  name of reference channel

**`-o`**  output directory

**`-n`**  multiprocessing: number of processes, default 1

####Optional

**`--tile_size`**  size of a square tile, default 1000, which corresponds to 1000x1000px tile

**`--overlap`**  overlap between tiles, default 100

**`--num_pyr_lvl`**  number of pyramid levels. Default 3, 0 - will not use pyramids

**`--num_iter`**  number of registration iterations per pyramid level. Default 3, cannot be less than 1

### Example usage

**`python opt_flow_reg.py -i /path/to/iamge/stack/out.tif -c "DAPI" -o /path/to/output/dir/ -n 3`**


### Dependencies
`numpy tifffile opencv-contrib-python dask scikit-learn`

