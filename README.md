![](./logo.png)
 # Trata Sampling Methods Package

Trata offers a large number of general sampling strategies that can be used to explore parameter spaces or improve a model's predictive ability.

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
## Documentation
The documentation can be built from the `docs` directory using:

```bash
make html
```

Read the Docs coming soon.

## Contact Info

Trata maintainer can be reached at: olson59@llnl.gov

## Contributing

Contributing to Trata is relatively easy. Just send us a pull request. When you send your request, make develop the destination branch on the Trata repository.

Your PR must pass Trata's unit tests and documentation tests, and must be PEP 8 compliant. We enforce these guidelines with our CI process. To run these tests locally, and for helpful tips on git, see our [Contribution Guide](.github/workflows/CONTRIBUTING.md).

Trata's `develop` branch has the latest contributions. Pull requests should target `develop`, and users who want the latest package versions, features, etc. can use `develop`.


Contributions should be submitted as a pull request pointing to the `develop` branch, and must pass Trata's CI process; to run the same checks locally, use:

```bash
pytest tests/test_*.py
```

## Releases
See our [change log](CHANGELOG.md) for more details.

## Code of Conduct
Please note that Trata has a [Code of Conduct](.github/workflows/CODE_OF_CONDUCT.md). By participating in the Trata community, you agree to abide by its rules.

## License
Trata is distributed under the terms of the MIT license. All new contributions must be made under the MIT license. See LICENSE and NOTICE for details.

LLNL-CODE-838977
