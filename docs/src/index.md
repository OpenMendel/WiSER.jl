# WiSER.jl

`WiSER.jl` implements a regression method for modeling the within-subject variability of a longitudinal measurement. It stands for **wi**thin-**s**ubject variance **e**stimation by robust **r**egression. 

## Model

TODO: don't need to give this much details. Make it clear what kind of problems it solves (a diagram will be perfect) and cite preprint will be enough.

## Installation

This package requires Julia v1.0 or later, which can be obtained from https://julialang.org/downloads/ or by building Julia from the sources in the https://github.com/JuliaLang/julia repository.

The package has not yet been registered and must be installed using the repository location. Start Julia and use the ] key to switch to the package manager REPL

```{julia}
(@v1.4) Pkg> add https://github.com/OpenMendel/WiSER.jl
```

Use the backspace key to return to the Julia REPL.


```julia
# for this tutorial
using CSV, JuliaDB, Random, WiSER
```

    ┌ Info: Precompiling WiSER [2ff19380-1883-49fc-9d10-450face6b90c]
    └ @ Base loading.jl:1260


## Example data

The example dataset, `sbp.csv`, is contained in `data` folder of the package. It is a simulated datatset with 500 individuals, each having 9~11 observations. The outcome, systolic blood pressure (SBP), is a function of other covariates. Below we read in the data as a `DataFrame` using the [CSV package](https://juliadata.github.io/CSV.jl). WiSER.jl can take other data table objects, such as `IndexedTables` from the [JuliaDB](https://github.com/JuliaData/JuliaDB.jl) package.


```julia
filepath = normpath(joinpath(dirname(pathof(WiSER)), "../data/"))
df = CSV.read(filepath * "sbp.csv")
```

    ┌ Warning: `CSV.read(input; kw...)` is deprecated in favor of `DataFrame!(CSV.File(input; kw...))`
    └ @ CSV /Users/huazhou/.julia/packages/CSV/OM6FO/src/CSV.jl:40





<table class="data-frame"><thead><tr><th></th><th>id</th><th>sbp</th><th>agegroup</th><th>gender</th><th>bmi</th><th>meds</th><th>bmi_std</th></tr><tr><th></th><th>Int64</th><th>Float64</th><th>Float64</th><th>String</th><th>Float64</th><th>String</th><th>Float64</th></tr></thead><tbody><p>5,011 rows × 7 columns</p><tr><th>1</th><td>1</td><td>159.586</td><td>3.0</td><td>Male</td><td>23.1336</td><td>NoMeds</td><td>-1.57733</td></tr><tr><th>2</th><td>1</td><td>161.849</td><td>3.0</td><td>Male</td><td>26.5885</td><td>NoMeds</td><td>1.29927</td></tr><tr><th>3</th><td>1</td><td>160.484</td><td>3.0</td><td>Male</td><td>24.8428</td><td>NoMeds</td><td>-0.154204</td></tr><tr><th>4</th><td>1</td><td>161.134</td><td>3.0</td><td>Male</td><td>24.9289</td><td>NoMeds</td><td>-0.0825105</td></tr><tr><th>5</th><td>1</td><td>165.443</td><td>3.0</td><td>Male</td><td>24.8057</td><td>NoMeds</td><td>-0.185105</td></tr><tr><th>6</th><td>1</td><td>160.053</td><td>3.0</td><td>Male</td><td>24.1583</td><td>NoMeds</td><td>-0.72415</td></tr><tr><th>7</th><td>1</td><td>162.1</td><td>3.0</td><td>Male</td><td>25.2543</td><td>NoMeds</td><td>0.188379</td></tr><tr><th>8</th><td>1</td><td>163.153</td><td>3.0</td><td>Male</td><td>24.3951</td><td>NoMeds</td><td>-0.527037</td></tr><tr><th>9</th><td>1</td><td>166.675</td><td>3.0</td><td>Male</td><td>26.1514</td><td>NoMeds</td><td>0.935336</td></tr><tr><th>10</th><td>2</td><td>130.765</td><td>1.0</td><td>Male</td><td>22.6263</td><td>NoMeds</td><td>-1.99977</td></tr><tr><th>11</th><td>2</td><td>131.044</td><td>1.0</td><td>Male</td><td>24.7404</td><td>NoMeds</td><td>-0.239477</td></tr><tr><th>12</th><td>2</td><td>131.22</td><td>1.0</td><td>Male</td><td>25.3415</td><td>NoMeds</td><td>0.260949</td></tr><tr><th>13</th><td>2</td><td>131.96</td><td>1.0</td><td>Male</td><td>25.6933</td><td>NoMeds</td><td>0.553886</td></tr><tr><th>14</th><td>2</td><td>130.09</td><td>1.0</td><td>Male</td><td>21.7646</td><td>NoMeds</td><td>-2.71724</td></tr><tr><th>15</th><td>2</td><td>130.556</td><td>1.0</td><td>Male</td><td>23.7895</td><td>NoMeds</td><td>-1.03123</td></tr><tr><th>16</th><td>2</td><td>132.001</td><td>1.0</td><td>Male</td><td>26.9103</td><td>NoMeds</td><td>1.56716</td></tr><tr><th>17</th><td>2</td><td>131.879</td><td>1.0</td><td>Male</td><td>24.1153</td><td>NoMeds</td><td>-0.759929</td></tr><tr><th>18</th><td>2</td><td>131.609</td><td>1.0</td><td>Male</td><td>25.3372</td><td>NoMeds</td><td>0.257432</td></tr><tr><th>19</th><td>2</td><td>132.149</td><td>1.0</td><td>Male</td><td>23.7171</td><td>NoMeds</td><td>-1.09154</td></tr><tr><th>20</th><td>2</td><td>130.653</td><td>1.0</td><td>Male</td><td>25.5947</td><td>NoMeds</td><td>0.471793</td></tr><tr><th>21</th><td>3</td><td>145.655</td><td>2.0</td><td>Male</td><td>25.3645</td><td>NoMeds</td><td>0.280102</td></tr><tr><th>22</th><td>3</td><td>147.384</td><td>2.0</td><td>Male</td><td>26.6756</td><td>NoMeds</td><td>1.37179</td></tr><tr><th>23</th><td>3</td><td>146.558</td><td>2.0</td><td>Male</td><td>25.6001</td><td>NoMeds</td><td>0.476309</td></tr><tr><th>24</th><td>3</td><td>146.731</td><td>2.0</td><td>Male</td><td>26.3532</td><td>NoMeds</td><td>1.10337</td></tr><tr><th>25</th><td>3</td><td>143.037</td><td>2.0</td><td>Male</td><td>24.4092</td><td>NoMeds</td><td>-0.515285</td></tr><tr><th>26</th><td>3</td><td>144.845</td><td>2.0</td><td>Male</td><td>25.1193</td><td>NoMeds</td><td>0.075975</td></tr><tr><th>27</th><td>3</td><td>145.366</td><td>2.0</td><td>Male</td><td>25.5029</td><td>NoMeds</td><td>0.395354</td></tr><tr><th>28</th><td>3</td><td>145.506</td><td>2.0</td><td>Male</td><td>25.9668</td><td>NoMeds</td><td>0.781658</td></tr><tr><th>29</th><td>3</td><td>143.155</td><td>2.0</td><td>Male</td><td>24.9327</td><td>NoMeds</td><td>-0.0793522</td></tr><tr><th>30</th><td>3</td><td>146.147</td><td>2.0</td><td>Male</td><td>25.0029</td><td>NoMeds</td><td>-0.020953</td></tr><tr><th>&vellip;</th><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td><td>&vellip;</td></tr></tbody></table>



## Formulate model

First we will create a `WSVarLmmModel` object from the dataframe.

The `WSVarLmmModel()` function takes the following arguments: 

- `meanformula`: the formula for the mean fixed effects β (variables in X matrix).
- `reformula`: the formula for the mean random effects γ (variables in Z matrix).
- `wsvarformula`: the formula  for the within-subject variance fixed effects τ (variables in W matrix). 
- `idvar`: the id variable for groupings. 
- `datatable`: the datatable holding all of the data for the model. Can be a `DataFrame` or various types of tables such as an `IndexedTable`.
- `wtvar`: Optional argument of variable name holding subject-level weights in the `datatable`.

For documentation of the `WSVarLmmModel` function, type `?WSVarLmmModel` in Julia REPL.
```@docs
WSVarLmmModel
```

We will model sbp as a function of age, gender, and bmi_std. `bmi_std` is the centered and scaled `bmi`. The following commands fit the following model:

$\text{sbp}_{ij} = \beta_0 + \beta_1 \text{agegroup}_{ij} + \beta_2 \text{gender}_{ij} + \beta_3 \text{bmi}_{ij} + \gamma_{i0} + \gamma_{i1}\text{bmi} + \epsilon_{ij}$

$\epsilon_{ij}$ is distributed with mean 0 variance $\sigma^2_{\epsilon_{ij}}$

$\gamma_{i} = (\gamma_{i0}, \gamma_{i1})$ has mean **0** and variance $\Sigma_\gamma$

$\sigma^2_{\epsilon_{ij}} = exp(\tau_0 + \tau_1 \text{agegroup}_{ij} + \tau_2 \text{gender}_{ij} + \tau_3 \text{bmi}_{ij})$


```julia
vlmm = WSVarLmmModel(
    @formula(sbp ~ 1 + agegroup + gender + bmi_std + meds), 
    @formula(sbp ~ 1 + bmi_std), 
    @formula(sbp ~ 1 + agegroup + meds + bmi_std),
    :id, df);
```

The `vlmm` object has the appropriate data. We can use the `fit!()` function to fit the model.

## Fit model

Main arguments of the `fit!()` function are:
* `m::WSVarLmmModel`: The model to fit.
* `solver`: Non-linear programming solver to be used.
* `runs::Integer`: Number of weighted nonlinear least squares runs. Default is 2.

For a complete documentation, type `?WSVarLmmModel` in Julia REPL.
```@docs
fit!
```


```julia
WiSER.fit!(vlmm)
```

    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************
    
    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.166678, ‖ΔL‖ = 0.100999, status = Optimal, time(s) = 0.425396
    run = 2, ‖Δβ‖ = 0.005220, ‖Δτ‖ = 0.006748, ‖ΔL‖ = 0.048735, status = Optimal, time(s) = 0.354986





    
    Within-subject variance estimation by robust regression (WiSER)
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ───────────────────────────────────────────────────────────
                         Estimate  Std. Error       Z  Pr(>|Z|)
    ───────────────────────────────────────────────────────────
    β1: (Intercept)   106.308       0.14384    739.07    <1e-99
    β2: agegroup       14.9844      0.0633245  236.63    <1e-99
    β3: gender: Male   10.0749      0.100279   100.47    <1e-99
    β4: bmi_std         0.296424    0.0139071   21.31    <1e-99
    β5: meds: OnMeds  -10.1107      0.122918   -82.26    <1e-99
    τ1: (Intercept)    -2.5212      0.393792    -6.40    <1e-9
    τ2: agegroup        1.50759     0.135456    11.13    <1e-28
    τ3: meds: OnMeds   -0.435225    0.0621076   -7.01    <1e-11
    τ4: bmi_std         0.0052695   0.0224039    0.24    0.8140
    ───────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.00196    0.0181387
     "γ2: bmi_std"      0.0181387  0.000549357
    




The estimated coefficients and random effects covariance parameters can be retrieved by


```julia
coef(vlmm)
```




    9-element Array{Float64,1}:
     106.30828661757685
      14.984423626292854
      10.074886642511625
       0.2964238570056824
     -10.110677648545206
      -2.5211956122809442
       1.5075882029978345
      -0.4352249760976689
       0.00526950183272128



or individually


```julia
vlmm.β
```




    5-element Array{Float64,1}:
     106.30828661757685
      14.984423626292854
      10.074886642511625
       0.2964238570056824
     -10.110677648545206




```julia
vlmm.τ
```




    4-element Array{Float64,1}:
     -2.5211956122809442
      1.5075882029978345
     -0.4352249760976689
      0.00526950183272128




```julia
vlmm.Σγ
```




    2×2 Array{Float64,2}:
     1.00196    0.0181387
     0.0181387  0.000549357



The variance-covariance matrix of the estimated parameters (β, τ, Lγ) can be rerieved by


```julia
vlmm.vcov
```




    12×12 Array{Float64,2}:
      0.0206899    -0.00753187   -0.00618382   …  -0.000123531   0.0644858
     -0.00753187    0.00400999    0.000152994      4.07896e-5   -0.0194226
     -0.00618382    0.000152994   0.0100558        4.35497e-5   -0.0299542
      5.60981e-5   -4.80751e-5    0.000108448      8.06623e-6    0.00149567
     -0.00311952   -0.000362412   0.00122535      -7.1571e-5     0.0168424
     -0.00652959    0.00207365    0.00276734   …   0.00217472   -1.70443
      0.00229271   -0.000743467  -0.000951293     -0.000740359   0.58213
     -0.000719608   0.000263081   0.000294779      0.000197117  -0.152908
      3.10756e-5    1.70391e-5   -0.00011849      -5.50781e-5    0.0266044
      0.000166021  -3.24178e-6   -0.00011537       9.0954e-6    -0.00139559
     -0.000123531   4.07896e-5    4.35497e-5   …   7.84536e-5   -0.0244586
      0.0644858    -0.0194226    -0.0299542       -0.0244586    19.1312



## Tips for improving estimation

`fit!` may fail due to various reasons. Often it indicates ill-conditioned data or an inadequate model. Following strategies may improve the fit. 

### Standardize continuous predictors

In above example, we used the standardardized `bmi`. If we used the original `bmi` variable, the estimates of τ are instable, reflected by the large standard errors.


```julia
# using unscaled bmi causes ill-conditioning
vlmm_bmi = WSVarLmmModel(
    @formula(sbp ~ 1 + agegroup + gender + bmi + meds), 
    @formula(sbp ~ 1 + bmi), 
    @formula(sbp ~ 1 + agegroup + meds + bmi),
    :id, df);
WiSER.fit!(vlmm_bmi)
```

    run = 1, ‖Δβ‖ = 0.208950, ‖Δτ‖ = 0.445610, ‖ΔL‖ = 2.027674, status = Optimal, time(s) = 0.876796
    run = 2, ‖Δβ‖ = 0.032012, ‖Δτ‖ = 0.014061, ‖ΔL‖ = 0.780198, status = Optimal, time(s) = 0.987205





    
    Within-subject variance estimation by robust regression (WiSER)
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ────────────────────────────────────────────────────────────
                          Estimate  Std. Error       Z  Pr(>|Z|)
    ────────────────────────────────────────────────────────────
    β1: (Intercept)   100.131        0.319906   313.00    <1e-99
    β2: agegroup       14.9844       0.0633245  236.63    <1e-99
    β3: gender: Male   10.0749       0.100279   100.47    <1e-99
    β4: bmi             0.246808     0.0115793   21.31    <1e-99
    β5: meds: OnMeds  -10.1107       0.122918   -82.26    <1e-99
    τ1: (Intercept)    -2.63101     17.2804      -0.15    0.8790
    τ2: agegroup        1.50759      5.69286      0.26    0.7911
    τ3: meds: OnMeds   -0.435225     1.37021     -0.32    0.7508
    τ4: bmi             0.00438748   0.0281074    0.16    0.8760
    ────────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  0.484542    0.00557087
     "γ2: bmi"          0.00557087  0.000380843
    




### Increase `runs`

Increasing `runs` (default is 2) takes more computing resources but can be useful to get more precise estimates. If we set `runs=3` when using original `bmi` (ill-conditioned), the estimated τ are more accurate. The estimate of Σγ is still off though.


```julia
# improve estimates from ill-conditioned data by more runs
WiSER.fit!(vlmm_bmi, runs=3)
```

    run = 1, ‖Δβ‖ = 0.208950, ‖Δτ‖ = 0.445610, ‖ΔL‖ = 2.027674, status = Optimal, time(s) = 0.850399
    run = 2, ‖Δβ‖ = 0.032012, ‖Δτ‖ = 0.014061, ‖ΔL‖ = 0.780198, status = Optimal, time(s) = 1.060520
    run = 3, ‖Δβ‖ = 0.008059, ‖Δτ‖ = 0.099534, ‖ΔL‖ = 0.696869, status = Optimal, time(s) = 1.127406





    
    Within-subject variance estimation by robust regression (WiSER)
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ─────────────────────────────────────────────────────────────
                           Estimate  Std. Error       Z  Pr(>|Z|)
    ─────────────────────────────────────────────────────────────
    β1: (Intercept)   100.139         0.315745   317.15    <1e-99
    β2: agegroup       14.9839        0.0633172  236.65    <1e-99
    β3: gender: Male   10.0753        0.10027    100.48    <1e-99
    β4: bmi             0.246528      0.0114083   21.61    <1e-99
    β5: meds: OnMeds  -10.1109        0.122778   -82.35    <1e-99
    τ1: (Intercept)    -2.53158       0.866855    -2.92    0.0035
    τ2: agegroup        1.50917       0.031734    47.56    <1e-99
    τ3: meds: OnMeds   -0.436745      0.0513571   -8.50    <1e-16
    τ4: bmi             0.000277851   0.0363866    0.01    0.9939
    ─────────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  3.48717e-48  7.26846e-26
     "γ2: bmi"          7.26846e-26  0.00155716
    




### Try different nonlinear programming (NLP) solvers 

A different solver may remedy the issue. By default, `MiSER.jl` uses the [Ipopt](https://github.com/jump-dev/Ipopt.jl) solver, but it can use any solver that supports [MathProgBase.jl](https://github.com/JuliaOpt/MathProgBase.jl). Check documentation of `fit!` for commonly used NLP solvers. In our experience, [Knitro.jl](https://github.com/JuliaOpt/KNITRO.jl) works the best, but it is a commercial software.


```julia
# print Ipopt iterates for diagnostics
WiSER.fit!(vlmm, Ipopt.IpoptSolver(print_level=5, mehrotra_algorithm="yes"))
```

    This is Ipopt version 3.13.2, running with linear solver mumps.
    NOTE: Other linear solvers might be more efficient (see Ipopt documentation).
    
    Number of nonzeros in equality constraint Jacobian...:        0
    Number of nonzeros in inequality constraint Jacobian.:        0
    Number of nonzeros in Lagrangian Hessian.............:       28
    
    Total number of variables............................:        7
                         variables with only lower bounds:        0
                    variables with lower and upper bounds:        0
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  5.6575142e+04 0.00e+00 5.30e+04   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  3.4931954e+04 0.00e+00 1.43e+04 -11.0 2.07e+00    -  1.00e+00 1.00e+00f  1
       2  3.0285692e+04 0.00e+00 4.18e+03 -11.0 5.46e-01    -  1.00e+00 1.00e+00f  1
       3  2.9124181e+04 0.00e+00 1.87e+03 -11.0 2.78e-01    -  1.00e+00 1.00e+00f  1
       4  2.8571986e+04 0.00e+00 6.41e+02 -11.0 2.38e-01    -  1.00e+00 1.00e+00f  1
       5  2.8379415e+04 0.00e+00 1.62e+02 -11.0 2.91e-01    -  1.00e+00 1.00e+00f  1
       6  2.8328064e+04 0.00e+00 4.31e+01 -11.0 3.09e-01    -  1.00e+00 1.00e+00f  1
       7  2.8315452e+04 0.00e+00 1.13e+01 -11.0 3.02e-01    -  1.00e+00 1.00e+00f  1
       8  2.8312446e+04 0.00e+00 4.59e+00 -11.0 2.76e-01    -  1.00e+00 1.00e+00f  1
       9  2.8311807e+04 0.00e+00 1.87e+00 -11.0 2.28e-01    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      10  2.8311705e+04 0.00e+00 5.50e-01 -11.0 1.52e-01    -  1.00e+00 1.00e+00f  1
      11  2.8311697e+04 0.00e+00 7.11e-02 -11.0 6.21e-02    -  1.00e+00 1.00e+00f  1
      12  2.8311697e+04 0.00e+00 1.34e-03 -11.0 8.86e-03    -  1.00e+00 1.00e+00f  1
      13  2.8311697e+04 0.00e+00 1.53e-05 -11.0 1.62e-04    -  1.00e+00 1.00e+00f  1
      14  2.8311697e+04 0.00e+00 2.18e-07 -11.0 8.29e-08    -  1.00e+00 1.00e+00f  1
      15  2.8311697e+04 0.00e+00 1.13e-08 -11.0 3.75e-10    -  1.00e+00 1.00e+00f  1
      16  2.8311697e+04 0.00e+00 3.05e-10 -11.0 2.12e-11    -  1.00e+00 1.00e+00f  1
    
    Number of Iterations....: 16
    
                                       (scaled)                 (unscaled)
    Objective...............:   1.6226171160601305e+04    2.8311697021847409e+04
    Dual infeasibility......:   3.0454227901697596e-10    5.3137050315399392e-10
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
    Overall NLP error.......:   3.0454227901697596e-10    5.3137050315399392e-10
    
    
    Number of objective function evaluations             = 17
    Number of objective gradient evaluations             = 17
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 16
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.031
    Total CPU secs in NLP function evaluations           =      0.346
    
    EXIT: Optimal Solution Found.
    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.166678, ‖ΔL‖ = 0.100999, status = Optimal, time(s) = 0.330199
    This is Ipopt version 3.13.2, running with linear solver mumps.
    NOTE: Other linear solvers might be more efficient (see Ipopt documentation).
    
    Number of nonzeros in equality constraint Jacobian...:        0
    Number of nonzeros in inequality constraint Jacobian.:        0
    Number of nonzeros in Lagrangian Hessian.............:       28
    
    Total number of variables............................:        7
                         variables with only lower bounds:        0
                    variables with lower and upper bounds:        0
                         variables with only upper bounds:        0
    Total number of equality constraints.................:        0
    Total number of inequality constraints...............:        0
            inequality constraints with only lower bounds:        0
       inequality constraints with lower and upper bounds:        0
            inequality constraints with only upper bounds:        0
    
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
       0  8.8973092e+04 0.00e+00 2.21e+05   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  4.3429572e+04 0.00e+00 6.17e+04 -11.0 2.38e+00    -  1.00e+00 1.00e+00f  1
       2  4.2299630e+04 0.00e+00 5.28e+04 -11.0 5.57e-01    -  1.00e+00 1.00e+00f  1
       3  3.2423451e+04 0.00e+00 1.96e+04 -11.0 5.34e-01    -  1.00e+00 1.00e+00f  1
       4  2.8893365e+04 0.00e+00 6.24e+03 -11.0 2.85e-01    -  1.00e+00 1.00e+00f  1
       5  2.7767774e+04 0.00e+00 2.56e+03 -11.0 2.67e-01    -  1.00e+00 1.00e+00f  1
       6  2.7349272e+04 0.00e+00 8.84e+02 -11.0 2.79e-01    -  1.00e+00 1.00e+00f  1
       7  2.7218041e+04 0.00e+00 2.32e+02 -11.0 3.08e-01    -  1.00e+00 1.00e+00f  1
       8  2.7182533e+04 0.00e+00 5.14e+01 -11.0 3.24e-01    -  1.00e+00 1.00e+00f  1
       9  2.7173294e+04 0.00e+00 1.62e+01 -11.0 3.27e-01    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      10  2.7170863e+04 0.00e+00 8.25e+00 -11.0 3.25e-01    -  1.00e+00 1.00e+00f  1
      11  2.7170227e+04 0.00e+00 4.19e+00 -11.0 3.19e-01    -  1.00e+00 1.00e+00f  1
      12  2.7170063e+04 0.00e+00 2.08e+00 -11.0 3.09e-01    -  1.00e+00 1.00e+00f  1
      13  2.7170022e+04 0.00e+00 9.97e-01 -11.0 2.89e-01    -  1.00e+00 1.00e+00f  1
      14  2.7170013e+04 0.00e+00 4.38e-01 -11.0 2.53e-01    -  1.00e+00 1.00e+00f  1
      15  2.7170011e+04 0.00e+00 1.56e-01 -11.0 1.91e-01    -  1.00e+00 1.00e+00f  1
      16  2.7170011e+04 0.00e+00 3.31e-02 -11.0 1.04e-01    -  1.00e+00 1.00e+00f  1
      17  2.7170011e+04 0.00e+00 1.88e-03 -11.0 2.70e-02    -  1.00e+00 1.00e+00f  1
      18  2.7170011e+04 0.00e+00 1.49e-05 -11.0 1.57e-03    -  1.00e+00 1.00e+00f  1
      19  2.7170011e+04 0.00e+00 5.09e-07 -11.0 5.31e-06    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      20  2.7170011e+04 0.00e+00 9.99e-09 -11.0 1.26e-08    -  1.00e+00 1.00e+00f  1
    
    Number of Iterations....: 20
    
                                       (scaled)                 (unscaled)
    Objective...............:   2.7170011141753250e+04    2.7170011141753250e+04
    Dual infeasibility......:   9.9916355189577644e-09    9.9916355189577644e-09
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
    Overall NLP error.......:   9.9916355189577644e-09    9.9916355189577644e-09
    
    
    Number of objective function evaluations             = 21
    Number of objective gradient evaluations             = 21
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 20
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.020
    Total CPU secs in NLP function evaluations           =      0.358
    
    EXIT: Optimal Solution Found.
    run = 2, ‖Δβ‖ = 0.005220, ‖Δτ‖ = 0.006748, ‖ΔL‖ = 0.048735, status = Optimal, time(s) = 0.384719





    
    Within-subject variance estimation by robust regression (WiSER)
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ───────────────────────────────────────────────────────────
                         Estimate  Std. Error       Z  Pr(>|Z|)
    ───────────────────────────────────────────────────────────
    β1: (Intercept)   106.308       0.14384    739.07    <1e-99
    β2: agegroup       14.9844      0.0633245  236.63    <1e-99
    β3: gender: Male   10.0749      0.100279   100.47    <1e-99
    β4: bmi_std         0.296424    0.0139071   21.31    <1e-99
    β5: meds: OnMeds  -10.1107      0.122918   -82.26    <1e-99
    τ1: (Intercept)    -2.5212      0.393792    -6.40    <1e-9
    τ2: agegroup        1.50759     0.135456    11.13    <1e-28
    τ3: meds: OnMeds   -0.435225    0.0621076   -7.01    <1e-11
    τ4: bmi_std         0.0052695   0.0224039    0.24    0.8140
    ───────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.00196    0.0181387
     "γ2: bmi_std"      0.0181387  0.000549357
    





```julia
# use Knitro (require installation of Knitro software and Knitro.jl)
# WiSER.fit!(vlmm, KNITRO.KnitroSolver(outlev=3));
```


```julia
# use NLopt
WiSER.fit!(vlmm, NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000))
```

    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.162196, ‖ΔL‖ = 0.100050, status = Optimal, time(s) = 0.571036
    run = 2, ‖Δβ‖ = 0.005248, ‖Δτ‖ = 0.008742, ‖ΔL‖ = 0.001334, status = Optimal, time(s) = 0.185684





    
    Within-subject variance estimation by robust regression (WiSER)
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ────────────────────────────────────────────────────────────
                          Estimate  Std. Error       Z  Pr(>|Z|)
    ────────────────────────────────────────────────────────────
    β1: (Intercept)   106.308        0.14384    739.07    <1e-99
    β2: agegroup       14.9844       0.0633238  236.63    <1e-99
    β3: gender: Male   10.0749       0.100277   100.47    <1e-99
    β4: bmi_std         0.296421     0.0139114   21.31    <1e-99
    β5: meds: OnMeds  -10.1106       0.122912   -82.26    <1e-99
    τ1: (Intercept)    -2.53263      0.102706   -24.66    <1e-99
    τ2: agegroup        1.51161      0.0388869   38.87    <1e-99
    τ3: meds: OnMeds   -0.435897     0.0524849   -8.31    <1e-16
    τ4: bmi_std         0.00576945   0.0218517    0.26    0.7918
    ────────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.00228    0.0179118
     "γ2: bmi_std"      0.0179118  0.00441753
    




### Try different starting points

Initialization matters as well. By default, `fit!` uses a crude least squares estimate as the starting point. We can also try a method of moment estimate or user-supplied values.


```julia
# MoM starting point
WiSER.fit!(vlmm, init = init_mom!(vlmm))
```

    run = 1, ‖Δβ‖ = 0.036245, ‖Δτ‖ = 0.188207, ‖ΔL‖ = 0.127483, status = Optimal, time(s) = 0.256069
    run = 2, ‖Δβ‖ = 0.006798, ‖Δτ‖ = 0.009128, ‖ΔL‖ = 0.050049, status = Optimal, time(s) = 0.340028





    
    Within-subject variance estimation by robust regression (WiSER)
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ────────────────────────────────────────────────────────────
                          Estimate  Std. Error       Z  Pr(>|Z|)
    ────────────────────────────────────────────────────────────
    β1: (Intercept)   106.308        0.143831   739.12    <1e-99
    β2: agegroup       14.9846       0.063327   236.62    <1e-99
    β3: gender: Male   10.0747       0.100282   100.46    <1e-99
    β4: bmi_std         0.296596     0.013989    21.20    <1e-99
    β5: meds: OnMeds  -10.1107       0.122973   -82.22    <1e-99
    τ1: (Intercept)    -2.52233      0.218068   -11.57    <1e-30
    τ2: agegroup        1.5079       0.0759423   19.86    <1e-87
    τ3: meds: OnMeds   -0.434951     0.0549139   -7.92    <1e-14
    τ4: bmi_std         0.00527178   0.0220323    0.24    0.8109
    ────────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.00193    0.0180064
     "γ2: bmi_std"      0.0180064  0.000967577
    





```julia
# user-supplied starting point in vlmm.β, vlmm.τ, vlmm.Lγ
# WiSER.fit!(vlmm, init = vlmm)
```

## Simulating responses

The `rvarlmm()` and `rvarlmm!()` functions can be used to generate a respone from user-supplied data and parameters. The `rand!()` command can be used to overwrite the response in a VarLmmModel object based on the parameters and optional user-supplied distribution.   

The `rand!(m::WSVarLmmModel; respdist = MvNormal, γωdist = MvNormal, Σγω = [], kwargs...)` function replaces the responses `m.data[i].y` with a simulated response based on:

- The data in the model object's data `X, Z, W` matrices. 
- The parameter values in the model.
- The condistribution distribution of the response given the random effects.
- The distribution of the random effects.
- If simulating from MvTDistribution, you must specify the degrees of freedom via `df = x`.

The `rvarlmm()` takes arrays of matricies of the data in addition to the reponse. It generates a simulated response from the VarLMM model based on:
- `Xs`: array of each clusters `X`: mean fixed effects covariates
- `Zs`: array of each clusters `Z`: random location effects covariates
- `Ws`: array of each clusters `W`: within-subject variance fixed effects covariates
- `β`: mean fixed effects vector
- `τ`: within-subject variance fixed effects vector
- `respdist`: the distribution for response. Default is MvNormal. 
- `Σγ`: random location effects covariance matrix. 
- `Σγω`: joint random location and random scale effects covariance matrix (if generating from full model).
- If simulating from MvTDistribution, you must specify the degrees of freedom via `df = x`.


The `rvarlmm!()` function can be used to generate a simulated response from the VarLMM model based on a datatable and place the generated response into the datatable with the `respname` field. 

Note: **the datatable MUST be ordered by grouping variable for it to generate in the correct order.**
This can be checked via `datatable == sort(datatable, idvar)`. The response is based on:

- `meanformula`: represents the formula for the mean fixed effects `β` (variables in X matrix)
- `reformula`: represents the formula for the mean random effects γ (variables in Z matrix)
- `wsvarformula`: represents the formula for the within-subject variance fixed effects τ (variables in W matrix)
- `idvar`: the id variable for groupings.
- `datatable`: the data table holding all of the data for the model. For this function it **must be in order**.
- `β`: mean fixed effects vector
- `τ`: within-subject variance fixed effects vector
- `respdist`: the distribution for response. Default is MvNormal. 
- `Σγ`: random location effects covariance matrix. 
- `Σγω`: joint random location and random scale effects covariance matrix (if generating from full model)
- `respname`: symbol representing the simulated response variable name.
- If simulating from MvTDistribution, you must specify the degrees of freedom via `df = x`.


For both functions, only one of the Σγ or Σγω matrices have to be specified in order to use the function. Σγ can be used to specify that the generative model will not include a random scale component. It outputs `ys`: an array of reponse `y` that match the order of the data arrays (`Xs, Zs, Ws`).


```julia
@show vlmm.data[1].y
Random.seed!(123)
WiSER.rand!(vlmm; respdist = MvNormal) 
@show vlmm.data[1].y
```

    (vlmm.data[1]).y = [159.58635186701068, 161.84850248386945, 160.48359574389164, 161.13448128282593, 165.44341004850986, 160.05302471176626, 162.1001598920002, 163.1526453898974, 166.6749897477845]
    (vlmm.data[1]).y = [163.18878816959145, 161.92583955740076, 160.66341989866248, 165.16516161135553, 162.25415689993756, 163.00335025501857, 162.06896794235755, 161.4110226386001, 160.57277004398432]





    9-element Array{Float64,1}:
     163.18878816959145
     161.92583955740076
     160.66341989866248
     165.16516161135553
     162.25415689993756
     163.00335025501857
     162.06896794235755
     161.4110226386001
     160.57277004398432




```julia
t = table((id = [1; 1; 2; 3; 3; 3; 4], y = randn(7),
x1 = ones(7), x2 = randn(7), x3 = randn(7), z1 = ones(7),
z2 = randn(7), w1 = ones(7), w2 = randn(7), w3 = randn(7)))
df = DataFrame(t)

f1 = @formula(y ~ 1 + x2 + x3)
f2 = @formula(y ~ 1 + z2)
f3 = @formula(y ~ 1 + w2 + w3)

β = zeros(3)
τ = zeros(3)
Σγ = [1. 0.; 0. 1.]

first(df, 3)
```




<table class="data-frame"><thead><tr><th></th><th>id</th><th>y</th><th>x1</th><th>x2</th><th>x3</th><th>z1</th><th>z2</th><th>w1</th><th>w2</th></tr><tr><th></th><th>Int64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th></tr></thead><tbody><p>3 rows × 10 columns (omitted printing of 1 columns)</p><tr><th>1</th><td>1</td><td>1.54783</td><td>1.0</td><td>0.268209</td><td>-0.0571664</td><td>1.0</td><td>0.222619</td><td>1.0</td><td>1.17759</td></tr><tr><th>2</th><td>1</td><td>0.365508</td><td>1.0</td><td>1.17666</td><td>-0.204264</td><td>1.0</td><td>1.0579</td><td>1.0</td><td>0.431064</td></tr><tr><th>3</th><td>2</td><td>-0.31447</td><td>1.0</td><td>0.453137</td><td>-0.402403</td><td>1.0</td><td>-1.65662</td><td>1.0</td><td>-0.216927</td></tr></tbody></table>




```julia
rvarlmm!(f1, f2, f3, :id, df, β, τ;
        Σγ = Σγ, respname = :response)
df[!, :response]
```




    7-element Array{Float64,1}:
     -1.1362274345153838
     -2.0396426949163917
     -3.0297957994022724
      0.7967897326514723
      1.1225160002462085
      1.2517509664063533
      0.34088047862482207



Note: JuliaDB's `IndexedTables` elements cannot be mutated. If you use the `rvarlmm!()` function with a JuliaDB table, you must reassign the `datatable` to the output as shown below:


```julia
t = rvarlmm!(f1, f2, f3, :id, t, β, τ;
        Σγ = Σγ, respname = :response)
```




    Table with 7 rows, 11 columns:
    Columns:
    [1m#   [22m[1mcolname   [22m[1mtype[22m
    ─────────────────────
    1   id        Int64
    2   y         Float64
    3   x1        Float64
    4   x2        Float64
    5   x3        Float64
    6   z1        Float64
    7   z2        Float64
    8   w1        Float64
    9   w2        Float64
    10  w3        Float64
    11  response  Float64


