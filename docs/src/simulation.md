# Simulating responses

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


The `rvarlmm!()` function can be used to generate a simulated response from the VarLMM model based on a dataframe and place the generated response into the datatable with the `respname` field. 

Note: **the dataframe MUST be ordered by grouping variable for it to generate in the correct order.**
This can be checked via `dataframe == sort(dataframe, idvar)`. The response is based on:

- `meanformula`: represents the formula for the mean fixed effects `β` (variables in X matrix)
- `reformula`: represents the formula for the mean random effects γ (variables in Z matrix)
- `wsvarformula`: represents the formula for the within-subject variance fixed effects τ (variables in W matrix)
- `idvar`: the id variable for groupings.
- `dataframe`: the dataframe holding all of the data for the model. For this function it **must be in order**.
- `β`: mean fixed effects vector
- `τ`: within-subject variance fixed effects vector
- `respdist`: the distribution for response. Default is MvNormal. 
- `Σγ`: random location effects covariance matrix. 
- `Σγω`: joint random location and random scale effects covariance matrix (if generating from full model)
- `respname`: symbol representing the simulated response variable name.
- If simulating from MvTDistribution, you must specify the degrees of freedom via `df = x`.


For both functions, only one of the Σγ or Σγω matrices have to be specified in order to use the function. Σγ can be used to specify that the generative model will not include a random scale component. It outputs `ys`: an array of reponse `y` that match the order of the data arrays (`Xs, Zs, Ws`).

We can start by loading the pacakges, data, and fitting a model.


```julia
using CSV, DataFrames, Random, WiSER
filepath = normpath(joinpath(dirname(pathof(WiSER)), "../data/"))
df = DataFrame!(CSV.File(filepath * "sbp.csv"))
vlmm = WSVarLmmModel(
    @formula(sbp ~ 1 + agegroup + gender + bmi_std + meds), 
    @formula(sbp ~ 1 + bmi_std), 
    @formula(sbp ~ 1 + agegroup + meds + bmi_std),
    :id, df);
WiSER.fit!(vlmm)
```

    
    ******************************************************************************
    This program contains Ipopt, a library for large-scale nonlinear optimization.
     Ipopt is released as open source code under the Eclipse Public License (EPL).
             For more information visit http://projects.coin-or.org/Ipopt
    ******************************************************************************
    
    run = 1, ‖Δβ‖ = 0.037311, ‖Δτ‖ = 0.166678, ‖ΔL‖ = 0.100999, status = Optimal, time(s) = 0.376659
    run = 2, ‖Δβ‖ = 0.005220, ‖Δτ‖ = 0.006748, ‖ΔL‖ = 0.048735, status = Optimal, time(s) = 0.234782





    
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
    




Once the model has been fit, we can overwrite the response variable simulating a new response based on the model's current parameters. This is done by calling the `rand!()` function on the model object. Here we simulate from a multivariate normal dsitribution for $y$.


```julia
yoriginal = copy(vlmm.data[1].y)
Random.seed!(123)
WiSER.rand!(vlmm; respdist = MvNormal) 
[yoriginal vlmm.data[1].y]
```




    9×2 Array{Float64,2}:
     159.586  163.223
     161.849  161.898
     160.484  160.667
     161.134  165.167
     165.443  162.258
     160.053  163.019
     162.1    162.065
     163.153  161.422
     166.675  160.552



Other response distributions have been coded. To get a list of available distributions use `respdists()`


```julia
respdists()
```




    6-element Array{Symbol,1}:
     :MvNormal
     :MvTDist
     :Gamma
     :InverseGaussian
     :InverseGamma
     :Uniform




```julia
WiSER.rand!(vlmm; respdist = InverseGamma) 
vlmm.data[1].y
```




    9-element Array{Float64,1}:
     167.83510676083995
     161.04081244800372
     161.88509094798928
     162.76369002769596
     168.02717792311043
     164.52117964053977
     162.84533339184907
     162.54354236314282
     163.87154251671376



We can also simulate a response variable from a dataframe and a formula. 

If you don't want to overwrite the response variable in the dataframe, you can use the `respname` optional keyword argument to specify the desired variable name to save the response variable as. 


```julia
df = DataFrame(id = [1; 1; 2; 3; 3; 3; 4], y = randn(7),
x2 = randn(7), x3 = randn(7), z2 = randn(7), w2 = randn(7), w3 = randn(7))

f1 = @formula(y ~ 1 + x2 + x3)
f2 = @formula(y ~ 1 + z2)
f3 = @formula(y ~ 1 + w2 + w3)

β = zeros(3)
τ = zeros(3)
Σγ = [1. 0.; 0. 1.]
rvarlmm!(f1, f2, f3, :id, df, β, τ;
        Σγ = Σγ, respname = :response)
[df[!, :y] df[!, :response]]
```




    7×2 Array{Float64,2}:
      1.54783   -1.13623
      0.365508  -2.03964
     -0.31447   -3.0298
     -0.291651   0.79679
      1.0763     1.12252
     -0.672566   1.25175
     -0.70343    0.34088


