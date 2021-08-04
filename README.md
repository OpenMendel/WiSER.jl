# WiSER.jl

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://openmendel.github.io/WiSER.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://openmendel.github.io/WiSER.jl/dev/) | [![Build Status](https://github.com/OpenMendel/WiSER.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/OpenMendel/WiSER.jl/actions/workflows/ci.yml)  | [![codecov](https://codecov.io/gh/OpenMendel/WiSER.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/OpenMendel/WiSER.jl) |  

WiSER stands for **wi**thin-**s**ubject variance **e**stimation by **r**obust regression. It is a regression aproach for estimating the effects of predictors on the within-subject variation in a longitudinal setting. 

WiSER.jl requires Julia v1.0 or later. See documentation for usage. It is not yet registered and can be installed, in the Julia Pkg mode, by
```{julia}
(@v1.6) Pkg> add https://github.com/OpenMendel/WiSER.jl
```

## Citation

The methods and applications of this software package are detailed in the following publication:

German CA, Sinsheimer JS, Zhou JJ., and Zhou H. (2021) WiSER: Robust and scalable estimation and inference of within-subject variances from intensive longitudinal data. Biometrics, in press. <https://doi.org/10.1111/biom.13506>

If you use [OpenMendel](https://openmendel.github.io) analysis packages in your research, please cite the following reference in the resulting publications:

Zhou H, Sinsheimer JS, Bates DM, Chu BB, German CA, Ji SS, Keys KL, Kim J, Ko S, Mosher GD, Papp JC, Sobel EM, Zhai J, Zhou JJ, Lange K. (2020) OPENMENDEL: a cooperative programming project for statistical genetics. Hum Genetics, 139(1):61-71. <https://doi.org/10.1007/s00439-019-02001-z> PMCID: PMC6763373

