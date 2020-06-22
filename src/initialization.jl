"""
    init_ls!(m::VarLmmModel)

Initialize parameters of a `VarLmmModel` object from least squares estimate. 
`m.β`  is initialized to be `inv(sum(xi'xi)) * sum(xi'yi)`. 
`m.τ`  is initialized to be `inv(sum(wi'wi)) * sum(wi'log(abs2(ri)))`.  
`m.Σγ` is initialized to be `inv(sum(zi'zi⊗zi'zi)) * sum(zi'ri⊗zi'ri)`. 
"""
function init_ls!(m::VarLmmModel{T}) where T <: BlasReal
    q = m.q
    # LS estimate for β
    fill!(m.∇β , 0) # to accumulate Xi'yi
    fill!(m.Hββ, 0) # to accumulate Xi'Xi
    for obs in m.data
        # Xi'Xi
        BLAS.syrk!('U', 'N', T(1), obs.Xt, T(1), m.Hββ)
        # Xi'yi
        BLAS.gemv!('N', T(1), obs.Xt, obs.y, T(1), m.∇β)
    end
    _, info = LAPACK.potrf!('U', m.Hββ)
    info > 0 && throw("design matrix X is singular")
    LAPACK.potrs!('U', m.Hββ, copyto!(m.β, m.∇β))
    update_res!(m)
    # accumulate quantities for initilizing Σγ and τ
    fill!(m.∇τ   , 0) # to accumulate Wi' * log(ri.^2)
    fill!(m.Hττ  , 0) # to accumulate Wi' * Wi
    fill!(m.∇Σγ  , 0) # to accumulate Zi'ri ⊗ Zi'ri
    fill!(m.HΣγΣγ, 0) # to accumulate Zi'Zi ⊗ Zi'Zi
    for obs in m.data
        # storage_n1 = log(diag(rr'))
        obs.storage_n1 .= log.(obs.res2)
        # accumulate Wi' * log(ri.^2)
        BLAS.gemv!('N', T(1), obs.Wt, obs.storage_n1, T(1), m.∇τ)
        # accumulate Wi' * Wi
        BLAS.syrk!('U', 'N', T(1), obs.Wt, T(1), m.Hττ)
        # Zi'Zi ⊗ Zi'Zi
        kron_axpy!(obs.ztz, obs.ztz, m.HΣγΣγ)
        # Zi'ri ⊗ Zi'ri
        kron_axpy!(obs.ztres, obs.ztres, vec(m.∇Σγ))
    end
    # LS estimate for Σγ
    _, info = LAPACK.potrf!('U', m.HΣγΣγ)
    info > 0 && throw("design matrix Z is singular")
    LAPACK.potrs!('U', m.HΣγΣγ, vec(copyto!(m.Σγ, m.∇Σγ)))
    _, info = LAPACK.potrf!('L', copy!(m.Lγ, m.Σγ))
    @inbounds for j in 2:q, i in 1:j-1 # make upper triangular of Lγ zero
        m.Lγ[i, j] = 0
    end
    # Σγ is singular; set columns L[:, info:end] = 0
    if info > 0
        @warn("starting Lγ is rank deficient")
        for j in info:q, i in j:q
            m.Lγ[i, j] = 0
        end
    end
    # regress log(ri .* ri) on Wi to initialize τ
    _, info = LAPACK.potrf!('U', m.Hττ)
    info > 0 && throw("design matrix W is singular")
    LAPACK.potrs!('U', m.Hττ, copyto!(m.τ, m.∇τ))
    m
end

"""
    init_mom!(m::VarLmmModel)

Initialize `τ` and `Lγ` of a `VarLmmModel` object by method of moment (MoM) 
using residulas in `m.obs[i].res`. It involves solving an unweighted NLS problem.

# Position arguments
- `m::VarLmmModel`: A `VarLmmModel` object.
- `solver`: Default is `Ipopt.IpoptSolver(print_level=5, watchdog_shortened_iter_trigger=3)`.

# Keyword arguments
- `init`: Initlizer for the NLS problem. Default is `init_ls!(m)`. If `init = m`, 
then it uses the values provided in `m.τ` and `m.Lγ` as starting point.
"""
function init_mom!(
    m::VarLmmModel{T},
    solver = Ipopt.IpoptSolver(print_level=5, watchdog_shortened_iter_trigger=3);
    init = init_ls!(m)
    ) where T <: BlasReal
    # set up NLP optimization problem
    npar = m.l + ◺(m.q)
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, npar)
    ub = fill( Inf, npar)
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Min, m)
    # optimize unweighted obj function (MoM estimator)
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, m)
    MathProgBase.setwarmstart!(optm, par0)
    m.iswtnls[1] = false
    MathProgBase.optimize!(optm)
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    optstat = MathProgBase.status(optm) 
    optstat == :Optimal || 
    @warn("Optimization unsuccesful; got $optstat")
    m
end