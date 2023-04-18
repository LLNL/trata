![](./logo.png)
 # Trata Bayesian Sampling

In creating a surrogate model, generating initial training data requires the selection of samples from the design parameter spaces. Trata is used to generate sample points in order to explore a parameter space. 

For instance, if a simulation takes two inputs, x and y, and you want to run a set of simulations with x-values between 5 and 20 and y-values between 0.1 and 1000, the sampling component can generate sample points (in this case (x,y) pairs) for you. 

You can specify how many total sample points you want, and how you want them to be chosen--Trata offers a large number of different sampling strategies. If, on the other hand, you already have sample points you wish to use, Trata can simply read them in from a file. 

Trata contains 3 modules:
   - **`composite_samples`**
   - **`sampler`**
   - **`adaptive_samples`**<br>
<br>

## `composite_samples`

The **`composite_samples`** module enables a user to parse a tab or csv file and create a "variable", or parameter, class object that represents discrete discrete-ordered, or continuous samples. The `parse_file` function returns a _`Samples`_ object containing the points from the file. Other file types would need to be parsed with a custom function. 

## `sampler`

The **`sampler`** module enables a user to select the type of sampling method they would like to perform across a design parameter space.  The available options include:
   - `CartesianCross` 
   - `Centered`
   - `Corner`
   - `Dakota`
   - `DefaultValue`
   - `Face`
   - `LatinHyperCube`
   - `MonteCarlo`
   - `MultiNormal`
   - `OneAtATime`
   - `ProbabilityDensityFunction`
   - `QuasiRandomNumber`
   - `Rejection`
   - `SamplePoint`
   - `Uniform`
   - `UserValue` <br>
<br>

## `adaptive_samples`

The number of samples required to build an accurate surrogate model is _a posteriori_ knowledge determined by the complexity of the approximated input-output relation. Therefore enriching the training dataset as training progresses is performed and is known as active learning. 

The **`adaptive_sampler`** module allows a user to specify learning functions to help identify the next sample with the highest information value. Those learning functions are designed to allocate samples to regions where the surrogate model is thought to be inaccurate or uncertain, or the regions where particularly interesting combinations of design parameters lie, such as the region that possibly contains the globally optimum values of the design parameters. The available options include:
   - `Scored`
   - `Weighted`
   - `ActiveLearning`
   - `Delta` 
   - `ExpectedImprovement`
   - `LearningExpectedImprovement`<br>
<br>

## Basic Installation

### via pip:

```bash
export TRATA_PATH = trata                     # `trata` can be any name/directory you want
pip install virtualenv                        # just in case
python3 -m virtualenv $TRATA_PATH   
source ${TRATA_PATH}/bin/activate
pip install numpy scikit-learn scipy matplotlib 
git clone https://github.com/LLNL/trata
cd trata
pip install .
```

### via conda:

```bash
conda create -n trata -c conda-forge "python>=3.6" numpy scikit-learn scipy matplotlib
conda activate trata
git clone https://github.com/LLNL/trata
cd trata
pip install .
```

## For Running Tests

```bash
pip install pytest 
```
### via conda:

```bash
conda install -n trata -c conda-forge pytest 
```

## For Building Docs

### via pip:

```bash
pip install sphinx sphinx_rtd_theme nbconvert sphinx-autoapi nbsphinx 
```
### via conda:

```bash
conda install -n trata -c conda-forge sphinx sphinx_rtd_theme nbconvert sphinx-autoapi nbsphinx
```

## Beefy Installation

### via pip:

```bash
export TRATA_PATH = trata                 # `trata` can be any name/directory you want
pip install virtualenv                    # just in case
python3 -m virtualenv $TRATA_PATH   
source ${TRATA_PATH}/bin/activate
pip install numpy scikit-learn scipy matplotlib six pip pytest sphinx sphinx_rtd_theme nbconvert sphinx-autoapi nbsphinx jupyterlab ipython ipywidgets 
git clone https://github.com/LLNL/trata
cd trata
pip install .
```
### via conda:

```bash
conda create -n trata -c conda-forge "python>=3.6" numpy scikit-learn scipy matplotlib six pip pytest pytest-json-report sphinx sphinx_rtd_theme nbconvert sphinx-autoapi nbsphinx jupyterlab ipython ipywidgets nb_conda nb_conda_kernels 
conda activate trata
git clone https://github.com/LLNL/trata
cd trata
pip install .
```

## Register your Python env via Jupyter:

```bash
python -m ipykernel install --user --name trata --display-name "Trata Environment"
```
Standard Setup
==============

Standard installation, provided by the Makefile, is initiated by entering
the following at the command line:

    $ make

This command creates the virtual environment, installs \(missing\) dependencies,
and installs Trata.

Test are run by entering:

    $ make run_tests

You can build the documentation from `docs`, which will appear in `build/docs`, using:

    $ make html
