# 1.0.3

# Release Notes

* [1.1.0](#110-release)
* [1.0.3](#103-release)
* [1.0.2](#102-release)
* [1.0.1](#101-release)


## 1.1.0 Release

### Description

This release introduces a new kosh_sampler module

### New in this release

* The kosh_sampler module wraps the adaptive sampling functions in a Kosh operator. Users with existing Kosh datasets can easily find the next best set of samples based on their model's error or sensitivity. 

## 1.0.3 Release

This is a minor release with a few bug fixes.

### Bug fixes
* fixed adaptive sampling tests to be more robust to numpy versions.
* fixed hanging problem in composite samples.
* A few other minor changes related to testing

## 1.0.2 Release

### Description

This release is a minor release with a few bux fixes and new features. We encourage users to upgrade.

### New in this release

* Added `**kwargs` to Morris sampler to catch extra arguments.

### Improvements
* Added code of conduct contributing documents


### Bug fixes
* No bug fixes


## 1.0.1 Release

### Description

This release is a minor release with a few bux fixes and new features. We encourage users to upgrade.
