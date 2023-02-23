I haven't been able to test this much, but outside of spatial constrains, this seems to work decently well, so it should be useful for LUT generation already. Outputting images is too slow for it to be of much interest, anyway.

# Moving Least Squares Color Transfer

Use moving least squares to generate color transferring 3D LUTs or transferred images.

Based on:
Hwang, Y., Lee, J. Y., Kweon, I. S., & Kim, S. J. (2019). Probabilistic moving least squares with spatial constraints for nonlinear color transfer between images. Computer Vision and Image Understanding, 180, 1-12. https://doi.org/10.1016/j.cviu.2018.11.001

### Differences to paper

* No probabilistic modeling is performed, as no image registration is necessary for our use case.
* Extrapolation just checks if nearest neighbor is within a sphere around the bin center. This is faster and should be good enough.
* No GPU processing (yet?). (Mine's too old.) I hope this is the reason processing is way slower than the paper states.

# Usage

```
usage: MLSColorTransfer [-S] [-P] [-s SPATIAL-WEIGHT]
                        [-r COLOR-WEIGHT] [-p PREFILTER-STRENGTH]
                        [-c CONTROL-POINTS] [-l LUT-SIZE] [-i IMAGE]
                        [-h] source target filename

Moving least squares for color transfer.

positional arguments:
  source                Source image(s). For multiple images, this
                        should be a folder with shared filenaming
                        between source and target images.
  target                Target image(s). For multiple images, this
                        should be a folder.with shared filenaming
                        between source and target images.
  filename              Output image (png) or 3D LUT (cube) file name.

optional arguments:
  -S, --spatial         Enable using spatial info. Can only be used if
                        outputting an image. Experimental.
  -P, --no-prefilter    Disable prefilter.
  -s, --spatial-weight SPATIAL-WEIGHT
                        Spatial domain weighting strength. (type:
                        Float64, default: 20.0)
  -r, --color-weight COLOR-WEIGHT
                        Color space domain weighting strength. (type:
                        Float64, default: 20.0)
  -p, --prefilter-strength PREFILTER-STRENGTH
                        Prefilter strength. (type: Float64, default:
                        2.0)
  -c, --control-points CONTROL-POINTS
                        Number of control points to use. Lowering this
                        will lead to worse results and potential
                        crashes, but will decrease computation time.
                        It's recommended to set to 1% of total pixel
                        count for best results. (type: Int64, default:
                        1000)
  -l, --lut-size LUT-SIZE
                        3D LUT size. (type: Int64, default: 33)
  -i, --image IMAGE     Image to apply transfer to. Must be set if
                        outputting an image.
  -h, --help            show this help message and exit
```

Some things can use multi threading. You can pass `--julia-args -t auto` to take full advantage of this. However, you might run out of memory.

Images need to be cropped and resized to match more or less exactly. Misaligned images will lead to terrible results.

Spatial information doesn't really work yet.
