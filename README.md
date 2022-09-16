# Sampling Methods

LLNL's Sampling Methods is used to generate sample points in order to explore a parameter space. For instance, if a simulation takes two inputs, x and y, and you want to run a set of simulations with x-values between 5 and 20 and y-values between 0.1 and 1000, the sampling component can generate sample points (which in this case means (x,y) pairs) for you. You can specify how many total sample points you want, and how you want them to be chosen--the sampling component offers a large number of different sampling strategies. If, on the other hand, you already have sample points you wish to use, the component can simply read them in from a file. 


The Sampling Methods contains 3 modules:
   - sampler
   - composite_samples
   - adaptive_samples


## Basic Installation

### via pip:

```bash
export SAMPLING_METHODS_PATH = sampling-methods                     # `sampling-methods` can be any name/directory you want
pip install virtualenv                                              # just in case
python3 -m virtualenv $SAMPLING_METHODS_PATH   
source ${SAMPLING_METHODS_PATH}/bin/activate
pip install "numpy>=1.15,<1.19" scikit-learn scipy matplotlib
git clone https://github.com/LLNL/sampling-methods
cd sampling-methods
pip install .
```

### via conda:

```bash
conda create -n sampling-methods -c conda-forge "python>=3.6" "numpy>=1.15,<1.19" scikit-learn scipy matplotlib
conda activate sampling-methods
git clone https://github.com/LLNL/sampling-methods
cd sampling-methods
pip install .
```
## Build Docs

### via pip:

```bash
pip install sphinx sphinx_rtd_theme sphinx-autoapi nbsphinx
```
### via conda:

```bash
conda install -n sampling-methods -c conda-forge sphinx sphinx_rtd_theme sphinx-autoapi nbsphinx
```

## Beefy Installation

### via pip:

```bash
export SAMPLING_METHODS_PATH = sampling-methods                 # `sampling-methods` can be any name/directory you want
pip install virtualenv                                          # just in case
python3 -m virtualenv $SAMPLING_METHODS_PATH   
source ${SAMPLING_METHODS_PATH}/bin/activate
pip install "numpy>=1.15,<1.19" scikit-learn scipy matplotlib six pip sphinx sphinx_rtd_theme sphinx-autoapi ipython jupyterlab nbsphinx ipywidgets 
git clone https://github.com/LLNL/sampling-methods
cd sampling-methods
pip install .
```
### via conda:

```bash
conda create -n sampling-methods -c conda-forge "python>=3.6" "numpy>=1.15,<1.19" scikit-learn scipy matplotlib six pip sphinx sphinx_rtd_theme sphinx-autoapi nbsphinx jupyterlab ipython ipywidgets nb_conda nb_conda_kernels 
conda activate sampling-methods
git clone https://github.com/LLNL/sampling-methods
cd sampling-methods
pip install .
```

### Register your Python env via Jupyter:

```bash
python -m ipykernel install --user --name sampling_methods --display-name "Sampling Methods Environment"
```