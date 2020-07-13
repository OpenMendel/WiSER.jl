# WiSER.jl

| **Documentation** | **Build Status** | **Code Coverage**  |
|-------------------|------------------|--------------------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://github.com/OpenMendel/WiSER.jl/stable) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://github.com/OpenMendel/WiSER.jl/dev) | [![Build Status](https://travis-ci.com/OpenMendel/WiSER.jl.svg?branch=master)](https://travis-ci.org/OpenMendel/WiSER.jl)  | [![Coverage Status](https://coveralls.io/repos/github/OpenMendel/WiSER.jl/badge.svg?branch=master)](https://coveralls.io/github/OpenMendel/WiSER.jl?branch=master) [![codecov](https://codecov.io/gh/OpenMendel/WiSER.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/OpenMendel/WiSER.jl) |  

WiSER stands for **wi**thin-**s**ubject variance **e**stimation by **r**obust regression. It is a regression aproach for estimating the effects of predictors on the within-subject variation in a longitudinal setting. 

WiSER.jl requires Julia v1.0 or later. See documentation for usage. It is not yet registered and can be installed, in the Julia Pkg mode, by
```{julia}
(@v1.4) Pkg> add https://github.com/OpenMendel/WiSER.jl
```
