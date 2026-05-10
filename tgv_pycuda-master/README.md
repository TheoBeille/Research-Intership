# PyCUDA primal-dual algorithms for TV/TGV-constrained imaging problems

[![DOI](https://zenodo.org/badge/986192133.svg)](https://zenodo.org/badge/latestdoi/986192133)

Algorithms, examples and tests for denoising, deblurring, zooming, dequantization and compressive imaging with total variation (TV) and second-order total generalized variation (TGV) regularization. Python implementation with GPU acceleration using PyCUDA.

The code reproduces, in particular, the numerical experiments in the associated publication:

> Kristian Bredies. Recovering piecewise smooth multichannel images by minimization of convex functionals with total generalized variation penalty. *Lecture Notes in Computer Science*, 8293:44-77, 2014. doi:[10.1007/978-3-642-54774-4_3](https://doi.org/10.1007/978-3-642-54774-4_3)
 
## Getting started

One easy way of getting started is to create a Python virtual environment, install the dependencies and to call a test script. For instance, run in the project folder:

```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
python test_denoise.py
```

Please note that a working CUDA installation is required, in particular, a CUDA-enabled GPU. The test scripts are best run in an interactive environment such as `ipython` or `jupyter-notebook`.

```
test_denoise.py
test_denoise2.py
test_denoise3.py
test_deblur.py
test_deblur2.py
test_zoom.py
test_zoom2.py
test_dequantize.py
test_compressed_sensing.py
```

## Guided examples and figures

A Jupyter Notebook is available that guides through the examples and reproduces the figures in the above-mentioned publication.

```bash
jupyter-notebook examples.ipynb
```

## Author

* **[Kristian Bredies](https://imsc.uni-graz.at/bredies/)**, [Department of Mathematics and Scientific Computing](https://mathematik.uni-graz.at/en), [University of Graz](https://www.uni-graz.at/en), kristian.bredies@uni-graz.at

 ## Acknowledgements
 
Support by the [Austrian Science Fund (FWF)](https://www.fwf.ac.at/en/) under grant [SFB F32](https://dx.doi.org/10.55776/F32) (Mathematical Optimization and Applications in Biomedical Sciences) is gratefully acknowledged.

If you use this code, please cite the associated publication:

> Kristian Bredies. Recovering piecewise smooth multichannel images by minimization of convex functionals with total generalized variation penalty. *Lecture Notes in Computer Science*, 8293:44-77, 2014. doi:[10.1007/978-3-642-54774-4_3](https://doi.org/10.1007/978-3-642-54774-4_3)

```bibtex
@inbook{Bredies2014,
  title = {Recovering Piecewise Smooth Multichannel Images by Minimization of Convex Functionals with Total Generalized Variation Penalty},
  DOI = {10.1007/978-3-642-54774-4_3},
  booktitle = {Efficient Algorithms for Global Optimization Methods in Computer Vision},
  publisher = {Springer Berlin Heidelberg},
  author = {Bredies, Kristian},
  year = {2014},
  pages = {44–77}
}
```

## License

This software, excluding third-party components, is licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE) for details.
