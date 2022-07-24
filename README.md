# WiSER.jl

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://openmendel.github.io/WiSER.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://openmendel.github.io/WiSER.jl/dev/) | [![Build Status](https://github.com/OpenMendel/WiSER.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/OpenMendel/WiSER.jl/actions/workflows/ci.yml)  | [![codecov](https://codecov.io/gh/OpenMendel/WiSER.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/OpenMendel/WiSER.jl) |  

WiSER stands for **wi**thin-**s**ubject variance **e**stimation by **r**obust regression. It is a regression aproach for estimating the effects of predictors on the within-subject variation in a longitudinal setting. 

WiSER.jl requires Julia v1.0 or later. See documentation for usage. I
This package is registered in the default Julia package registry, and can be installed through standard package installation procedure: e.g., running the following code in Julia REPL.
```julia
using Pkg
pkg"add WiSER"
```

## Citation

The methods and applications of this software package are detailed in the following publication:

*German CA, Sinsheimer JS, Zhou J, Zhou H. WiSER: Robust and scalable estimation and inference of within-subject variances from intensive longitudinal data. Biometrics. 2021 Jun 18:10.1111/biom.13506. doi: 10.1111/biom.13506. Epub ahead of print. PMID: 34142722; PMCID: PMC8683571.*

If you use [OpenMendel](https://openmendel.github.io) analysis packages in your research, please cite the following reference in the resulting publications:

*Zhou H, Sinsheimer JS, Bates DM, Chu BB, German CA, Ji SS, Keys KL, Kim J, Ko S, Mosher GD, Papp JC, Sobel EM, Zhai J, Zhou JJ, Lange K. OPENMENDEL: a cooperative programming project for statistical genetics. Hum Genet. 2020 Jan;139(1):61-71. doi: 10.1007/s00439-019-02001-z. Epub 2019 Mar 26. PMID: 30915546; PMCID: [PMC6763373](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6763373/).*

### Acknowledgments

This project has been supported by the National Institutes of Health under awards R01GM053275, R01HG006139, R25GM103774, and 1R25HG011845.
