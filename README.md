# HORCL : Robust Multi-robot Collaborative Localization in High Outlier ratio Scenes

## Overview
This is a c++ implementation based on [strasdat/Sophus](https://github.com/strasdat/Sophus). The solvers' cpp files are in the **test/ceres/** directory.

## Dependency

  - Eigen 3.3.4
  - Google ceres 2.0.0

## Quickstart
Assume the current directory is the root of this repository.

> Compile
```sh
$ chmod +x ./scripts/run_cpp_tests.sh
$ ./scripts/run_cpp_tests.sh
```

> Run
```sh
$ chmod +x ./scripts/run_robust_pcl_reconstruction_example_cauchy_two_EM.sh
$ ./scripts/run_robust_pcl_reconstruction_example_cauchy_two_EM.sh
```

## Common problems
1. **ccache** may not be installed by default. Simply install it.
```sh
$ sudo apt-get install ccache
```
