# Model Fitting

`WiSER.jl` implements a regression method for modeling the within-subject variability of a longitudinal measurement. It stands for **wi**thin-**s**ubject variance **e**stimation by robust **r**egression. 

Here we cover model construction and parameter estimation using WiSER.


```julia
using CSV, DataFrames, WiSER
```

## Example data

The example dataset, `sbp.csv`, is contained in `data` folder of the package. It is a simulated datatset with 500 individuals, each having 9 to 11 observations. The outcome, systolic blood pressure (SBP), is a function of other covariates. Below we read in the data as a `DataFrame` using the [CSV package](https://juliadata.github.io/CSV.jl). WiSER.jl can take other data table objects that comply with the `Tables.jl` format, such as `IndexedTables` from the [JuliaDB](https://github.com/JuliaData/JuliaDB.jl) package.


```julia
filepath = normpath(joinpath(dirname(pathof(WiSER)), "../data/"))
df = DataFrame(CSV.File(filepath * "sbp.csv"));
```

```jldoctest
5011×8 DataFrame
  Row │ id     sbp      agegroup  gender  bmi      meds    bmi_std     obswt   
      │ Int64  Float64  Float64   String  Float64  String  Float64     Float64 
──────┼────────────────────────────────────────────────────────────────────────
    1 │     1  159.586       3.0  Male    23.1336  NoMeds  -1.57733        4.0
    2 │     1  161.849       3.0  Male    26.5885  NoMeds   1.29927        4.0
    3 │     1  160.484       3.0  Male    24.8428  NoMeds  -0.154204       4.0
    4 │     1  161.134       3.0  Male    24.9289  NoMeds  -0.0825105      4.0
    5 │     1  165.443       3.0  Male    24.8057  NoMeds  -0.185105       4.0
    6 │     1  160.053       3.0  Male    24.1583  NoMeds  -0.72415        4.0
    7 │     1  162.1         3.0  Male    25.2543  NoMeds   0.188379       4.0
    8 │     1  163.153       3.0  Male    24.3951  NoMeds  -0.527037       4.0
  ⋮   │   ⋮       ⋮        ⋮        ⋮        ⋮       ⋮         ⋮          ⋮
 5005 │   500  155.672       3.0  Female  24.4651  NoMeds  -0.468741       3.0
 5006 │   500  148.389       3.0  Female  25.8129  NoMeds   0.653514       3.0
 5007 │   500  152.491       3.0  Female  24.5818  NoMeds  -0.371555       3.0
 5008 │   500  153.844       3.0  Female  25.721   NoMeds   0.57693        3.0
 5009 │   500  150.164       3.0  Female  24.3545  NoMeds  -0.560843       3.0
 5010 │   500  150.248       3.0  Female  23.8532  NoMeds  -0.978159       3.0
 5011 │   500  152.433       3.0  Female  26.1232  NoMeds   0.911814       3.0
```

## Formulate model

First we will create a `WSVarLmmModel` object from the dataframe.

The `WSVarLmmModel()` function takes the following arguments: 

- `meanformula`: the formula for the mean fixed effects β (variables in X matrix).
- `reformula`: the formula for the mean random effects γ (variables in Z matrix).
- `wsvarformula`: the formula  for the within-subject variance fixed effects τ (variables in W matrix). 
- `idvar`: the id variable for groupings. 
- `tbl`: the datatable holding all of the data for the model. Can be a `DataFrame` or various types of tables that comply with `Tables.jl` formatting, such as an `IndexedTable`.
- `wtvar`: Optional argument of variable name holding subject-level weights in the `tbl`.

For documentation of the `WSVarLmmModel` function, type `?WSVarLmmModel` in Julia REPL.
```@docs
WSVarLmmModel
```

We will model sbp as a function of age, gender, and bmi_std. `bmi_std` is the centered and scaled `bmi`. The following commands fit the following model:


$\text{sbp}_{ij} = \beta_0 + \beta_1 \cdot \text{agegroup}_{ij} + \beta_2 \cdot \text{gender}_{i} + \beta_3 \cdot \text{bmi}_{ij} + \gamma_{i0} + \gamma_{i1} \cdot \text{bmi} + \epsilon_{ij}$

``\epsilon_{ij}`` has mean 0 and variance ``\sigma^2_{\epsilon_{ij}}``

``\gamma_{i} = (\gamma_{i0}, \gamma_{i1})`` has mean **0** and variance ``\Sigma_\gamma``

$\sigma^2_{\epsilon_{ij}} = \exp(\tau_0 + \tau_1 \cdot \text{agegroup}_{ij} + \tau_2 \cdot \text{gender}_{i} + \tau_3 \cdot \text{bmi}_{ij})$


```julia
vlmm = WSVarLmmModel(
    @formula(sbp ~ 1 + agegroup + gender + bmi_std + meds), 
    @formula(sbp ~ 1 + bmi_std), 
    @formula(sbp ~ 1 + agegroup + meds + bmi_std),
    :id, df);
```

The `vlmm` object has the appropriate data formalated above. We can now use the `fit!()` function to fit the model.

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
             For more information visit https://github.com/coin-or/Ipopt
    ******************************************************************************
    
    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.166678, ‖ΔL‖ = 0.100999, status = Optimal, time(s) = 0.201188
    run = 2, ‖Δβ‖ = 0.005220, ‖Δτ‖ = 0.006748, ‖ΔL‖ = 0.048735, status = Optimal, time(s) = 0.080523





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi_std + meds
    Random Effects Formula:
    sbp ~ 1 + bmi_std
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi_std
    
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
    τ1: (Intercept)    -2.5212      0.393792    -6.40    <1e-09
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




    9-element Vector{Float64}:
     106.3082866175766
      14.984423626293006
      10.074886642511672
       0.29642385700569635
     -10.110677648545401
      -2.5211956122840613
       1.5075882029989467
      -0.43522497609297117
       0.005269501831413771



or individually


```julia
vlmm.β
```




    5-element Vector{Float64}:
     106.3082866175766
      14.984423626293006
      10.074886642511672
       0.29642385700569635
     -10.110677648545401




```julia
vlmm.τ
```




    4-element Vector{Float64}:
     -2.5211956122840613
      1.5075882029989467
     -0.43522497609297117
      0.005269501831413771




```julia
vlmm.Σγ
```




    2×2 Matrix{Float64}:
     1.00196    0.0181387
     0.0181387  0.000549357



The variance-covariance matrix of the estimated parameters (β, τ, Lγ) can be rerieved by


```julia
vlmm.vcov
```




    12×12 Matrix{Float64}:
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



Confidence intervals for $\boldsymbol{\beta}, \boldsymbol{\tau}$ can be obtained by `confint`. By default it returns 95% confidence intervals ($\alpha$ level = 0.05). 


```julia
confint(vlmm)
```




    9×2 Matrix{Float64}:
     106.026      106.59
      14.8603      15.1085
       9.87834     10.2714
       0.269167     0.323681
     -10.3516      -9.86976
      -3.29301     -1.74938
       1.2421       1.77308
      -0.556954    -0.313496
      -0.0386413    0.0491803




```julia
# 90% confidence interval
confint(vlmm, 0.1)
```




    9×2 Matrix{Float64}:
     106.29       106.326
      14.9765      14.9924
      10.0623      10.0875
       0.294676     0.298171
     -10.1261     -10.0952
      -2.57068     -2.47171
       1.49057      1.52461
      -0.44303     -0.42742
       0.0024542    0.00808481



**Note**: The default solver for WiSER.jl is :

`Ipopt.IpoptSolver(print_level=0, mehrotra_algorithm = "yes", warm_start_init_point="yes", max_iter=100)` 

This was chosen as it a free, open-source solver and the options typically reduce line search and lead to much faster fitting than other options. However, it can be a bit more instable. Below are tips to help improve estimation if the fit seems very off or fails. Switching the solver options or removing them and assigning it to the base Ipopt Solver `Ipopt.IpoptSolver(max_iter=100)` can take longer to converge but is usually a bit more stable. 

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

    run = 1, ‖Δβ‖ = 0.208950, ‖Δτ‖ = 0.445610, ‖ΔL‖ = 2.027674, status = Optimal, time(s) = 0.079164
    run = 2, ‖Δβ‖ = 0.032012, ‖Δτ‖ = 0.014061, ‖ΔL‖ = 0.780198, status = Optimal, time(s) = 0.125981





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi + meds
    Random Effects Formula:
    sbp ~ 1 + bmi
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi
    
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

    run = 1, ‖Δβ‖ = 0.208950, ‖Δτ‖ = 0.445610, ‖ΔL‖ = 2.027674, status = Optimal, time(s) = 0.085767
    run = 2, ‖Δβ‖ = 0.032012, ‖Δτ‖ = 0.014061, ‖ΔL‖ = 0.780198, status = Optimal, time(s) = 0.129032
    run = 3, ‖Δβ‖ = 0.008059, ‖Δτ‖ = 0.000678, ‖ΔL‖ = 0.083976, status = Optimal, time(s) = 0.154331





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi + meds
    Random Effects Formula:
    sbp ~ 1 + bmi
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi
    
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ────────────────────────────────────────────────────────────
                          Estimate  Std. Error       Z  Pr(>|Z|)
    ────────────────────────────────────────────────────────────
    β1: (Intercept)   100.139        0.315745   317.15    <1e-99
    β2: agegroup       14.9839       0.0633172  236.65    <1e-99
    β3: gender: Male   10.0753       0.10027    100.48    <1e-99
    β4: bmi             0.246528     0.0114083   21.61    <1e-99
    β5: meds: OnMeds  -10.1109       0.122778   -82.35    <1e-99
    τ1: (Intercept)    -2.63079      0.453424    -5.80    <1e-08
    τ2: agegroup        1.5079       0.0253371   59.51    <1e-99
    τ3: meds: OnMeds   -0.435791     0.051245    -8.50    <1e-16
    τ4: bmi             0.00436541   0.0178825    0.24    0.8071
    ────────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  0.377439    0.00949012
     "γ2: bmi"          0.00949012  0.000238614
    




### Try different nonlinear programming (NLP) solvers 

A different solver may remedy the issue. By default, `WiSER.jl` uses the [Ipopt](https://github.com/jump-dev/Ipopt.jl) solver, but it can use any solver that supports [MathProgBase.jl](https://github.com/JuliaOpt/MathProgBase.jl). Check documentation of `fit!` for commonly used NLP solvers. In our experience, [Knitro.jl](https://github.com/JuliaOpt/KNITRO.jl) works the best, but it is a commercial software.


```julia
# watchdog_shortened_iter_trigger option in IPOPT can sometimes be more robust to numerical issues
WiSER.fit!(vlmm, Ipopt.IpoptSolver(print_level=0, watchdog_shortened_iter_trigger=3, max_iter=100))
```

    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.166678, ‖ΔL‖ = 0.100999, status = Optimal, time(s) = 0.081864
    run = 2, ‖Δβ‖ = 0.005220, ‖Δτ‖ = 0.006748, ‖ΔL‖ = 0.048735, status = Optimal, time(s) = 0.068715





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi_std + meds
    Random Effects Formula:
    sbp ~ 1 + bmi_std
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi_std
    
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
    τ1: (Intercept)    -2.5212      0.393792    -6.40    <1e-09
    τ2: agegroup        1.50759     0.135456    11.13    <1e-28
    τ3: meds: OnMeds   -0.435225    0.0621076   -7.01    <1e-11
    τ4: bmi_std         0.0052695   0.0224039    0.24    0.8140
    ───────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.00196    0.0181387
     "γ2: bmi_std"      0.0181387  0.000549357
    





```julia
# print Ipopt iterates for diagnostics
WiSER.fit!(vlmm, Ipopt.IpoptSolver(print_level=5, mehrotra_algorithm="yes", warm_start_init_point="yes"))
```

    This is Ipopt version 3.13.4, running with linear solver mumps.
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
       0  2.8331778e+04 0.00e+00 1.00e+02   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  2.8314110e+04 0.00e+00 2.49e+01 -11.0 3.08e-01    -  1.00e+00 1.00e+00f  1
       2  2.8312135e+04 0.00e+00 3.32e+00 -11.0 2.63e-01    -  1.00e+00 1.00e+00f  1
       3  2.8311752e+04 0.00e+00 1.36e+00 -11.0 2.08e-01    -  1.00e+00 1.00e+00f  1
       4  2.8311700e+04 0.00e+00 3.34e-01 -11.0 1.25e-01    -  1.00e+00 1.00e+00f  1
       5  2.8311697e+04 0.00e+00 2.79e-02 -11.0 3.98e-02    -  1.00e+00 1.00e+00f  1
       6  2.8311697e+04 0.00e+00 2.40e-04 -11.0 3.48e-03    -  1.00e+00 1.00e+00f  1
       7  2.8311697e+04 0.00e+00 5.47e-06 -11.0 2.47e-05    -  1.00e+00 1.00e+00f  1
       8  2.8311697e+04 0.00e+00 9.63e-08 -11.0 1.20e-08    -  1.00e+00 1.00e+00f  1
       9  2.8311697e+04 0.00e+00 5.21e-09 -11.0 1.61e-10    -  1.00e+00 1.00e+00f  1
    
    Number of Iterations....: 9
    
                                       (scaled)                 (unscaled)
    Objective...............:   1.6226171160602307e+04    2.8311697021847336e+04
    Dual infeasibility......:   5.2103428765066918e-09    9.0910940997446232e-09
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
    Overall NLP error.......:   5.2103428765066918e-09    9.0910940997446232e-09
    
    
    Number of objective function evaluations             = 10
    Number of objective gradient evaluations             = 10
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 9
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.008
    Total CPU secs in NLP function evaluations           =      0.057
    
    EXIT: Optimal Solution Found.
    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.166678, ‖ΔL‖ = 0.100999, status = Optimal, time(s) = 0.081372
    This is Ipopt version 3.13.4, running with linear solver mumps.
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
       0  2.7170793e+04 0.00e+00 1.52e+01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
       1  2.7170194e+04 0.00e+00 3.80e+00 -11.0 3.15e-01    -  1.00e+00 1.00e+00f  1
       2  2.7170055e+04 0.00e+00 1.91e+00 -11.0 3.07e-01    -  1.00e+00 1.00e+00f  1
       3  2.7170020e+04 0.00e+00 9.10e-01 -11.0 2.86e-01    -  1.00e+00 1.00e+00f  1
       4  2.7170013e+04 0.00e+00 3.93e-01 -11.0 2.47e-01    -  1.00e+00 1.00e+00f  1
       5  2.7170011e+04 0.00e+00 1.35e-01 -11.0 1.82e-01    -  1.00e+00 1.00e+00f  1
       6  2.7170011e+04 0.00e+00 2.58e-02 -11.0 9.30e-02    -  1.00e+00 1.00e+00f  1
       7  2.7170011e+04 0.00e+00 1.16e-03 -11.0 2.12e-02    -  1.00e+00 1.00e+00f  1
       8  2.7170011e+04 0.00e+00 9.88e-06 -11.0 9.61e-04    -  1.00e+00 1.00e+00f  1
       9  2.7170011e+04 0.00e+00 2.64e-07 -11.0 2.10e-06    -  1.00e+00 1.00e+00f  1
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
      10  2.7170011e+04 0.00e+00 4.68e-09 -11.0 6.64e-09    -  1.00e+00 1.00e+00f  1
    
    Number of Iterations....: 10
    
                                       (scaled)                 (unscaled)
    Objective...............:   2.7170011141755771e+04    2.7170011141755771e+04
    Dual infeasibility......:   4.6827675070915120e-09    4.6827675070915120e-09
    Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
    Complementarity.........:   0.0000000000000000e+00    0.0000000000000000e+00
    Overall NLP error.......:   4.6827675070915120e-09    4.6827675070915120e-09
    
    
    Number of objective function evaluations             = 11
    Number of objective gradient evaluations             = 11
    Number of equality constraint evaluations            = 0
    Number of inequality constraint evaluations          = 0
    Number of equality constraint Jacobian evaluations   = 0
    Number of inequality constraint Jacobian evaluations = 0
    Number of Lagrangian Hessian evaluations             = 10
    Total CPU secs in IPOPT (w/o function evaluations)   =      0.008
    Total CPU secs in NLP function evaluations           =      0.057
    
    EXIT: Optimal Solution Found.
    run = 2, ‖Δβ‖ = 0.005220, ‖Δτ‖ = 0.006748, ‖ΔL‖ = 0.048735, status = Optimal, time(s) = 0.068690





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi_std + meds
    Random Effects Formula:
    sbp ~ 1 + bmi_std
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi_std
    
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
    τ1: (Intercept)    -2.5212      0.393792    -6.40    <1e-09
    τ2: agegroup        1.50759     0.135456    11.13    <1e-28
    τ3: meds: OnMeds   -0.435225    0.0621076   -7.01    <1e-11
    τ4: bmi_std         0.0052695   0.0224039    0.24    0.8140
    ───────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.00196    0.0181387
     "γ2: bmi_std"      0.0181387  0.000549357
    





```julia
# use Knitro (require installation of Knitro software and Knitro.jl)
# Using KNITRO
# WiSER.fit!(vlmm, KNITRO.KnitroSolver(outlev=3));
```


```julia
# use NLopt
WiSER.fit!(vlmm, NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000))
```

    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.162196, ‖ΔL‖ = 0.100050, status = Optimal, time(s) = 0.148150
    run = 2, ‖Δβ‖ = 0.005248, ‖Δτ‖ = 0.008747, ‖ΔL‖ = 0.001335, status = Optimal, time(s) = 0.052991





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi_std + meds
    Random Effects Formula:
    sbp ~ 1 + bmi_std
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi_std
    
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ───────────────────────────────────────────────────────────
                         Estimate  Std. Error       Z  Pr(>|Z|)
    ───────────────────────────────────────────────────────────
    β1: (Intercept)   106.308       0.14384    739.07    <1e-99
    β2: agegroup       14.9844      0.0633238  236.63    <1e-99
    β3: gender: Male   10.0749      0.100277   100.47    <1e-99
    β4: bmi_std         0.296421    0.0139114   21.31    <1e-99
    β5: meds: OnMeds  -10.1106      0.122912   -82.26    <1e-99
    τ1: (Intercept)    -2.53263     0.102707   -24.66    <1e-99
    τ2: agegroup        1.51161     0.038887    38.87    <1e-99
    τ3: meds: OnMeds   -0.435901    0.0524849   -8.31    <1e-16
    τ4: bmi_std         0.0057698   0.0218516    0.26    0.7917
    ───────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.00228    0.0179118
     "γ2: bmi_std"      0.0179118  0.00441744
    




Using a different solver can even help without the need for standardizing predictors. If we use the NLOPT solver with the `LD_MMA` algorithm on the model where bmi is not standardized we don't see heavily inflated standard errors.


```julia
# Using other solvers can work without standardizing 
WiSER.fit!(vlmm_bmi, NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=4000))
```

    run = 1, ‖Δβ‖ = 0.208950, ‖Δτ‖ = 0.143776, ‖ΔL‖ = 1.528229, status = Optimal, time(s) = 0.604965
    run = 2, ‖Δβ‖ = 0.026830, ‖Δτ‖ = 0.000125, ‖ΔL‖ = 0.000257, status = Optimal, time(s) = 0.046570





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi + meds
    Random Effects Formula:
    sbp ~ 1 + bmi
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi
    
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ───────────────────────────────────────────────────────────
                         Estimate  Std. Error       Z  Pr(>|Z|)
    ───────────────────────────────────────────────────────────
    β1: (Intercept)   100.126       0.323755   309.26    <1e-99
    β2: agegroup       14.9849      0.0633317  236.61    <1e-99
    β3: gender: Male   10.0748      0.10029    100.46    <1e-99
    β4: bmi             0.246967    0.0117298   21.05    <1e-97
    β5: meds: OnMeds  -10.1094      0.122977   -82.21    <1e-99
    τ1: (Intercept)    -3.01501     0.811039    -3.72    0.0002
    τ2: agegroup        1.50948     0.0468194   32.24    <1e-99
    τ3: meds: OnMeds   -0.426979    0.0519209   -8.22    <1e-15
    τ4: bmi             0.0192299   0.0368267    0.52    0.6016
    ───────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"   3.89413   -0.127903
     "γ2: bmi"          -0.127903   0.00561294
    




### Try different starting points

Initialization matters as well. By default, `fit!` uses a crude least squares estimate as the starting point. We can also try a method of moment estimate or user-supplied values.


```julia
# MoM starting point
WiSER.fit!(vlmm, init = init_mom!(vlmm))
```

    run = 1, ‖Δβ‖ = 0.036245, ‖Δτ‖ = 0.188207, ‖ΔL‖ = 0.127483, status = Optimal, time(s) = 0.062208
    run = 2, ‖Δβ‖ = 0.006798, ‖Δτ‖ = 0.009128, ‖ΔL‖ = 0.050049, status = Optimal, time(s) = 0.059064





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi_std + meds
    Random Effects Formula:
    sbp ~ 1 + bmi_std
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi_std
    
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
vlmm.β .= [106.0; 15.0; 10.0; 0.3; -10.0]
vlmm.τ .= [-2.5; 1.5; -0.5; 0.0]
vlmm.Lγ .= [1.0 0.0; 0.0 0.0]

fit!(vlmm, init = vlmm)
```

    run = 1, ‖Δβ‖ = 0.337743, ‖Δτ‖ = 0.069850, ‖ΔL‖ = 0.017323, status = Optimal, time(s) = 0.078268
    run = 2, ‖Δβ‖ = 0.003050, ‖Δτ‖ = 0.004463, ‖ΔL‖ = 0.001185, status = Optimal, time(s) = 0.104889





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi_std + meds
    Random Effects Formula:
    sbp ~ 1 + bmi_std
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi_std
    
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ────────────────────────────────────────────────────────────
                          Estimate  Std. Error       Z  Pr(>|Z|)
    ────────────────────────────────────────────────────────────
    β1: (Intercept)   106.309        0.143859   738.98    <1e-99
    β2: agegroup       14.984        0.0633192  236.64    <1e-99
    β3: gender: Male   10.0754       0.100275   100.48    <1e-99
    β4: bmi_std         0.296078     0.0136905   21.63    <1e-99
    β5: meds: OnMeds  -10.1108       0.122807   -82.33    <1e-99
    τ1: (Intercept)    -2.52144      0.0576657  -43.73    <1e-99
    τ2: agegroup        1.50787      0.0253351   59.52    <1e-99
    τ3: meds: OnMeds   -0.436135     0.0512042   -8.52    <1e-16
    τ4: bmi_std         0.00525556   0.0214765    0.24    0.8067
    ────────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.00203    0.0184422
     "γ2: bmi_std"      0.0184422  0.000339423
    




## Additional Features

WiSER.jl has additional features that may benefit some users. These include parallelizaiton and observation weights.

### Parallelization

WiSER.jl by default will not run objective function evaluations in parallel, but one at a time. In many cases (small number of individuals/relatively small number of observations per individual) it is faster to not parallelize the code as the internal overhead in setting up evaluations on multiple threads takes longer than the evaluations. However, with large numbers of observations per individual, or many individuals, it can be faster to parallelize.

In order to allow for parallelization, the julia environmental variable `JULIA_NUM_THREADS` should be set to a value greater than 1. This must be set before julia launches and can be done in couple ways:

- Setting a default number of threads for Julia to launch with in a `.bash_profile` file by adding a line `export JULIA_NUM_THREADS=X`. where X is the number of threads you wish to make the default.
- Before launching julia in the terminal, export the variable as done below:

```
$ export JULIA_NUM_THREADS=X
$ julia
```

This is different from the threads available used by BLAS commands. To check this number of threads for parallelization, run the following:


```julia
Threads.nthreads()
```




    4



We see there are 4 threads available. 

To parallelize the objective function in WiSER, simply add the keyword argument `parallel = true` in the `fit!()` function.


```julia
WiSER.fit!(vlmm, parallel = true)
```

    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.166678, ‖ΔL‖ = 0.100999, status = Optimal, time(s) = 0.237454
    run = 2, ‖Δβ‖ = 0.005220, ‖Δτ‖ = 0.006748, ‖ΔL‖ = 0.048735, status = Optimal, time(s) = 0.158717





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi_std + meds
    Random Effects Formula:
    sbp ~ 1 + bmi_std
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi_std
    
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
    τ1: (Intercept)    -2.5212      0.393792    -6.40    <1e-09
    τ2: agegroup        1.50759     0.135456    11.13    <1e-28
    τ3: meds: OnMeds   -0.435225    0.0621076   -7.01    <1e-11
    τ4: bmi_std         0.0052695   0.0224039    0.24    0.8140
    ───────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.00196    0.0181387
     "γ2: bmi_std"      0.0181387  0.000549357
    




We can see slight timing differences at this sample size:


```julia
@time WiSER.fit!(vlmm, parallel = false);
```

    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.166678, ‖ΔL‖ = 0.100999, status = Optimal, time(s) = 0.065098
    run = 2, ‖Δβ‖ = 0.005220, ‖Δτ‖ = 0.006748, ‖ΔL‖ = 0.048735, status = Optimal, time(s) = 0.070394
      0.141617 seconds (417 allocations: 38.531 KiB)



```julia
@time WiSER.fit!(vlmm, parallel = true);
```

    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.166678, ‖ΔL‖ = 0.100999, status = Optimal, time(s) = 0.150904
    run = 2, ‖Δβ‖ = 0.005220, ‖Δτ‖ = 0.006748, ‖ΔL‖ = 0.048735, status = Optimal, time(s) = 0.164348
      0.325590 seconds (1.95 k allocations: 174.953 KiB)


### Observation Weights

It can be useful for some users to fit WiSER with observation weights. We have implemented this feature, which can be done in the model constructor via the `wtvar` keyword. Note: Within each individual, observation weights are the same. We assume weights are per-indiviudal.

In the example data, the dataframe has a column `obswt`, corresponding to observation weights for each individual. 


```julia
vlmm_wts = WSVarLmmModel(
    @formula(sbp ~ 1 + agegroup + gender + bmi_std + meds), 
    @formula(sbp ~ 1 + bmi_std), 
    @formula(sbp ~ 1 + agegroup + meds + bmi_std),
    :id, df; wtvar = :obswt);

@time WiSER.fit!(vlmm_wts)
```

    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.158033, ‖ΔL‖ = 0.102058, status = Optimal, time(s) = 0.062162
    run = 2, ‖Δβ‖ = 0.006134, ‖Δτ‖ = 0.007594, ‖ΔL‖ = 0.056873, status = Optimal, time(s) = 0.078738
      0.146753 seconds (447 allocations: 40.500 KiB)





    
    Within-subject variance estimation by robust regression (WiSER)
    
    Mean Formula:
    sbp ~ 1 + agegroup + gender + bmi_std + meds
    Random Effects Formula:
    sbp ~ 1 + bmi_std
    Within-Subject Variance Formula:
    sbp ~ 1 + agegroup + meds + bmi_std
    
    Number of individuals/clusters: 500
    Total observations: 5011
    
    Fixed-effects parameters:
    ────────────────────────────────────────────────────────────
                          Estimate  Std. Error       Z  Pr(>|Z|)
    ────────────────────────────────────────────────────────────
    β1: (Intercept)   106.309       0.143971    738.41    <1e-99
    β2: agegroup       14.9841      0.0633336   236.59    <1e-99
    β3: gender: Male   10.0748      0.100288    100.46    <1e-99
    β4: bmi_std         0.296066    0.0139064    21.29    <1e-99
    β5: meds: OnMeds  -10.1101      0.122602    -82.46    <1e-99
    τ1: (Intercept)    -2.51639     0.267541     -9.41    <1e-20
    τ2: agegroup        1.50717     0.0914489    16.48    <1e-60
    τ3: meds: OnMeds   -0.445596    0.0366911   -12.14    <1e-33
    τ4: bmi_std         0.00634263  0.00812327    0.78    0.4349
    ────────────────────────────────────────────────────────────
    Random effects covariance matrix Σγ:
     "γ1: (Intercept)"  1.05449    0.0279852
     "γ2: bmi_std"      0.0279852  0.000792437
    



