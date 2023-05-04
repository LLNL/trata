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

## Getting Started

To get the latest public version:

```bash
pip install trata
```

To get the latest stable from a cloned repo, simply run:

```bash
pip install .
```

Alternatively, add the path to this repo to your PYTHONPATH environment variable or in your code with:

```bash
import sys
sys.path.append(path_to_trata_repo)
```
## Contact Info

Trata maintainer can be reached at: eljurf1@llnl.gov

## Contributing

Contributions should be submitted as a pull request pointing to the develop branch, and must pass Trata's CI process; to run the same checks locally, use:

```bash
pytest tests/test_*.py
```
