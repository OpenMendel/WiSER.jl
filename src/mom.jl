#functions for calculating and fitting method of moments
#estimates for VarLMM model

#vech operation
function vech(a::Union{Number, AbstractVecOrMat})
    m, n = size(a)
    out = similar(a, convert(Int, (2m - n + 1) * n / 2))
    ooffset, aoffset = 1, 1
    for j = 1:n
      len = m - j + 1 # no. elements to copy in column j
      copyto!(out, ooffset, a, aoffset, len)
      ooffset += m - j + 1
      aoffset += m + 1
    end
    out
end
# a is a scalar or (column) vector
vech(a::Union{Number, AbstractVector}) = copy(a)

function vech!(v::Union{Number, AbstractVector},
    a::Union{Number, AbstractVecOrMat},
    offset::Int = 0)
    m, n = size(a)
    m == n || error("Matrix input was not square.")
    ooffset, aoffset = 1, 1
    ooffset += offset
    for j = 1:n
      len = m - j + 1 # no. elements to copy in column j
      copyto!(v, ooffset, a, aoffset, len)
      ooffset += m - j + 1
      aoffset += m + 1
    end
end

#evaluates the variance constant
#e(Lγ, lγω, lω) = exp(1/2(lγω'lγω + lω^2 + 2lγω'Lγlγω + lγω'Lγ Lγ' lγω))
function varconst(Lγ, lγω, lω)
    v1 = BLAS.gemv('T', Lγ, lγω)
    v2 = BLAS.gemv('N', Lγ, lγω)
    return exp(0.5(sum(abs2, lγω) + lω[1]^2 + 2 * dot(lγω, v2) + sum(abs2, v1)))
end

"""
init_β_τ!(m)
Initialize the linear regression parameters `β` and `τ` by the least 
squares solutions.
"""
function init_β_τ!(
    m::varlmmModel{T}
    ) where T <: BlasReal
    # accumulate sufficient statistics X'y
    xty = zeros(T, m.p)
    xtx = zeros(T, m.p, m.p)
    meanw = zeros(T, m.l)
    wtlogvary = zeros(T, m.l)
    wtw = zeros(T, m.l, m.l)

    for i in eachindex(m.data)
        BLAS.gemv!('T', one(T), m.data[i].X, m.data[i].y, one(T), xty)
        BLAS.syrk!('U', 'T', one(T), m.data[i].X, one(T), xtx)
        meanw = mean(m.data[i].W, dims = [1])
        # or fill!(logvary, log(var(m.data[i].y)))
        #BLAS.gemv!('T', one(T), meanw, log(var(m.data[i].y)), one(T), wtlogvary)
        BLAS.axpy!(log(var(m.data[i].y)), meanw, wtlogvary)
        BLAS.syrk!('U', 'T', one(T), meanw, one(T), wtw)
    end
    # least square solution for β
    ldiv!(m.β, cholesky(Symmetric(xtx)), xty)
    # ldiv!(Y, A, B) -> Y
    # Compute A \ B in-place and store the result in Y, returning the result.
    # The argument A should not be a matrix. 
    # Rather, instead of matrices it should be a factorization object (e.g. produced by factorize or cholesky). 
    # The reason for this is that factorization itself is both expensive and typically allocates memory 
    # (although it can also be done in-place via, e.g., lu!), 
    # and performance-critical situations requiring ldiv! usually also require fine-grained control 
    # over the factorization of A.

    #estimate τ by W'W \ logvar(y)
    #average over W or replicate logvar y n times?
    ldiv!(m.τ, cholesky(Symmetric(wtw)), wtlogvary)
end

#gets indicies for diagonal elements of matrix's vech elements (ensure >= 0)
function diagInds(n::Int)
    inds = [1]
    for i in 1:(n - 1)
        ind = inds[i] + (n - i + 1)
        push!(inds, ind)
    end
    inds
end

"""
vec2ltri!(m::AbstractMatrix{T}, v::AbstractVector{T}, z::T = zero(T)) where T <: Real
Inplace storage of a vector to a lower triangular matrix. 

ex:
julia> mat = zeros(4, 4); x = 1.0:10.0
1.0:1.0:10.0

julia> vec2ltri!(mat, x)
4×4 Array{Float64,2}:
 1.0  0.0  0.0   0.0
 2.0  5.0  0.0   0.0
 3.0  6.0  8.0   0.0
 4.0  7.0  9.0  10.0
"""
function vec2ltri!(m::AbstractMatrix{T}, v::AbstractVector{T}, z::T = zero(T)) where T <: Real
    n = length(v)
    s = round(Int, sqrt(8n + 1) - 1) >> 1
    (s * (s + 1)) >> 1 == n || error("vec2ltri: length of vector is not triangular")
    offset = 1
    for j in 1:s
        for i in 1:s
            if i >= j 
                m[i, j] = v[offset]
                offset += 1
            else
                m[i, j] = z
            end
        end
    end
    m
end

"""
update_res!(obs, β)
Update the residual vector according to `β`.
"""
function update_res!(
    obs::varlmmObs{T}, 
    β::Vector{T}
    ) where T <: BlasReal
    #copyto!(obs.res, obs.y)
    for i in eachindex(obs.y)
        obs.res[i] = obs.y[i] - dot(view(obs.X, i, :), β)
    end
    #BLAS.gemv!('N', -one(T), obs.X, β, one(T), obs.res) # obs.res - obs.X * β = obs.y - obs.X * β
    #obs.res = obs.y - obs.X * β
    obs.res
end

function update_res!(
    m::varlmmModel{T}
    ) where T <: BlasReal
    for i in eachindex(m.data)
        update_res!(m.data[i], m.β)
    end
    nothing
end

"""
exp!(x::Array{T})

Inplace evaluator of elementwise exponential. 
ex:

julia> x = [log(1); log(2); log(3)]
3-element Array{Float64,1}:
 0.0               
 0.6931471805599453
 1.0986122886681098

julia> exp!(x)
3-element Array{Float64,1}:
 1.0
 2.0
 3.0
"""
function exp!(x::Array{T}) where T <: BlasReal
    for i in eachindex(x)
        x[i] = exp(x[i])
    end
end

"""
MoMobjf(obs::varlmmObs{T},
β::Vector{T},
τ::Vector{T},
Lγ::Matrix{T},
lγω::Vector{T},
lω::Vector{T};
eV::Float64 = Inf, #e(Lγ,lγω,lω) scalar part of Vi only uses parameters (no data)
needgrad::Bool = true
) where T <: BlasReal

MoMobjf!(
    m::varlmmModel{T},
    needgrad::Bool = true
    ) where T <: BlasReal

Evaluates the method of moments objective function for the given data and parameter estimates
    and also the gradient. eV can be precalculated to reduce computation time in fitting. 
"""
function MoMobjf!(obs::varlmmObs{T},
        β::Vector{T},
        τ::Vector{T},
        Lγ::Matrix{T},
        lγω::Vector{T},
        lω::Vector{T},
        eV::Float64 = Inf, #e(Lγ,lγω,lω) scalar part of Vi only uses parameters (no data)
        needgrad::Bool = true
        #y::Vector{Float64},
        #X::Matrix{Float64},
        #Z::Matrix{Float64},
        #W::Matrix{Float64},
        #inds::AbstractVector{<:Integer}
        ) where T <: BlasReal
        objvalue = 0.0
        n, p, q, l = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2), size(obs.W, 2)
        if needgrad
            fill!(obs.∇β, 0)
            fill!(obs.∇τ, 0)
            fill!(obs.∇Lγ, 0)
            fill!(obs.∇lγω, 0)
            fill!(obs.∇lω, 0)
        end

        #calculate residuals Yi - Xiβ and residual matrix
        update_res!(obs, β)
        #mul!(obs.storage_nn, obs.res, transpose(obs.res)) #may be faster with BLAS command, try BLAS.syrk? 
        BLAS.syrk!('U', 'N', one(T), obs.res, 0.0, obs.storage_nn)
        copytri!(obs.storage_nn, 'U')
        #calculate V_i
        # V = obs.Z * Σγ * obs.Z' + diag(exp(1/2 (σω^2) + w_{ij}^T * τ))
        # V = obs.Z * Lγ * Lγ' * obs.Z' + exp(1/2(lγω'lγω + lω^2 + 2lγω'Lγlγω + lγω'Lγ Lγ' lγω)) * diag(exp(w_{ij}^T * τ))
        
        #calculate e(Lγ, lγω, lω) = exp(1/2(lγω'lγω + lω^2 + 2lγω'Lγlγω + lγω'Lγ Lγ' lγω))
        # If not precalculated and input
        if eV == Inf
            eV = varconst(Lγ, lγω, lω)
        end
        #maybe pass this expression in since it contains no data, just parameters
        #then only have to calculate it once

        #calculate rest of Variance
        mul!(obs.storage_qn, transpose(Lγ), transpose(obs.Z))
        mul!(obs.V, transpose(obs.storage_qn), obs.storage_qn)
        mul!(obs.Wτ, obs.W, τ)
        exp!(obs.Wτ)
        for i in 1:n
            @views obs.V[i, i] += eV * obs.Wτ[i] 
        end

        #objective function is \sum_i ||(Yi - Xiβ)(Yi - Xiβ)^T - Vi||^2_F
        #BLAS.gemv!('N', 1, ) #to get R, the part inside the objective function 
        #BLAS.gemv!('T', one(T), m.data[i].X, m.data[i].y, one(T), xty)
        obs.storage_nn .-= obs.V #this is R_i
        # gemv!(tA, alpha, A, x, beta, y) 
        # Update the vector y as alpha*A*x + beta*y or alpha*A'x + beta*y according to tA. 
        # alpha and beta are scalars. Return the updated y.
        objvalue = 0.5 * sum(abs2, obs.storage_nn) # Is this most efficient way to get? 

        #gradient
        if needgrad
            # wrt β
            mul!(obs.storage_n1, obs.storage_nn, obs.res)
            mul!(obs.∇β, transpose(obs.X), obs.storage_n1)
            obs.∇β .*= -2 

            # wrt τ
            mul!(obs.storage_n1, Diagonal(obs.storage_nn), obs.Wτ)
            eVsumRiW = -eV * sum(obs.storage_n1) #need for gradient wrt Lγ
            mul!(obs.∇τ, transpose(obs.W), obs.storage_n1)
            obs.∇τ .*= -eV 

            # wrt Lγ
            # overwrite storage_qn 
            #BLAS.gemm!('T', 'N', one(T), obs.Z, obs.storage_nn, zero(T), obs.storage_qn)
            mul!(obs.storage_qn, transpose(obs.Z), obs.storage_nn)
            mul!(obs.storage_qq, obs.storage_qn, obs.Z)
            mul!(obs.storage_qq2, obs.storage_qq, Lγ)
            BLAS.syrk!('L', 'N', eVsumRiW, lγω, 0.0, obs.storage_qq) 
            copytri!(obs.storage_qq, 'L')
            BLAS.gemm!('N', 'N', 1.0, obs.storage_qq, I + Lγ, -2.0, obs.storage_qq2) 
            vech!(obs.∇Lγ, obs.storage_qq2)

            #BLAS.gemm!('T', 'N', one(T), obs.Z, obs.storage_nn, zero(T), obs.storage_qn)
            # BLAS.gemm!('N', 'N', one(T), obs.storage_qn, obs.Z, zero(T), obs.storage_qq)
            # BLAS.gemm!('N', 'N', one(T), obs.storage_qn, obs.Z, one(T), obs.storage_qq)
            # BLAS.gemm!('N', 'N', one(T), obs.storage_qq, Lγ, one(T), obs.storage_qq2)
            # BLAS.gemm!('N', 'T', eVsumRiW, lγω, lγω, zero(T), obs.storage_qq)
            # BLAS.gemm!('N', 'N', 1.0, obs.storage_qq, I + Lγ, -2.0, obs.storage_qq2)
            #BLAS.axpy!(one(T)) q x 1 improvement herer
            # vech!(obs.∇Lγ, obs.storage_qq2)

            # wrt lγω
            BLAS.gemv!('N', eVsumRiW, Lγ, lγω, one(T), obs.∇lγω)
            BLAS.gemv!('T', eVsumRiW, Lγ, lγω, one(T), obs.∇lγω)
            BLAS.gemm!('T', 'N', eVsumRiW, Lγ, Lγ, zero(T), obs.storage_qq)
            BLAS.gemv!('N', one(T), obs.storage_qq, lγω, one(T), obs.∇lγω)
            BLAS.axpy!(eVsumRiW, lγω, obs.∇lγω)

            # wrt lω
            BLAS.axpy!(eVsumRiW, lω, obs.∇lω)
        end
    return objvalue
end

"""
Update model Lγ from MixedModel's fit
"""
function extractLγ!(m::varlmmModel{T},
    mixedL::LinearAlgebra.LowerTriangular{T, Array{T,2}},
    sigma::T
    ) where T <: BlasReal
    for j in 1:m.q
        for i in 1:m.q
           j <= i ? m.Lγ[i, j] = sigma * mixedL[i, j] : continue
        end
    end
end

function MoMobjf!(
    m::varlmmModel{T},
    needgrad::Bool = true
    ) where T <: BlasReal
    objvalue = zero(T)
    if needgrad
        fill!(m.∇β, 0)
        fill!(m.∇τ, 0)
        fill!(m.∇Lγ, 0)
        fill!(m.∇lγω, 0)
        fill!(m.∇lω, 0)
    end
    #calculate e(Lγ, lγω, lω) = exp(1/2(lγω'lγω + lω^2 + 2lγω'Lγlγω + lγω'Lγ Lγ' lγω))
    #same for every observation (no data, only parameters)
    eV = varconst(m.Lγ, m.lγω, m.lω)
    for i in eachindex(m.data)
        objvalue += MoMobjf!(m.data[i], m.β, m.τ, m.Lγ, m.lγω, m.lω, eV, needgrad)
        if needgrad
            m.∇β .+= m.data[i].∇β
            m.∇τ .+= m.data[i].∇τ
            m.∇Lγ .+= m.data[i].∇Lγ
            m.∇lγω .+= m.data[i].∇lγω
            m.∇lω .+= m.data[i].∇lω
        end
    end
    objvalue
end



function fit!(
    m::varlmmModel,
    #solver=Ipopt.IpoptSolver(print_level=6)
    #solver=NLopt.NLoptSolver(algorithm=:LD_SLSQP, maxeval=10000)
    solver=NLopt.NLoptSolver(algorithm=:LD_MMA, maxeval=10000)
    )
    p, q, l = size(m.data[1].X, 2), size(m.data[1].Z, 2), size(m.data[1].W, 2)
    npar = p + l + ((q + 1) * (q + 2)) >> 1
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the q+1xq+1 random effect mx has ((q + 1) * (q + 2))/2 values
    #init_β_τ!(m) 
    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, npar) # error variance should be nonnegative, will fix later
    ub = fill(Inf, npar)
    copyto!(ub, p + l + 1 + (q * (q + 1)) >> 1, zeros(Float64, q))
    copyto!(lb, p + l + 1 + (q * (q + 1)) >>  1, zeros(Float64, q))

    lb[end] = -Inf
    ub[end] = -Inf
    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Min, m)
    # starting point
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, m)
    MathProgBase.setwarmstart!(optm, par0)
    #print("after setwarmstart, par0 = ", par0, "\n")
    # optimize
    MathProgBase.optimize!(optm)
    # print("after optimize!, getsolution(optm) = ", MathProgBase.getsolution(optm), "\n")
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # update parameters and refresh gradient
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    MoMobjf!(m, true)
    m
end

"""
    modelpar_to_optimpar!(m, par)
Translate model parameters in `m` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
    par::Vector,
    m::varlmmModel
    )
    p, q, l = size(m.data[1].X, 2), size(m.data[1].Z, 2), size(m.data[1].W, 2)
    copyto!(par, m.β)
    copyto!(par, p + 1, m.τ) 
    offset = p + l
    vech!(par, m.Lγ, offset)
    #change diagonals of m.Lγ to log() for unconstrained optimization
    nonnegInds = diagInds(q) .+ (offset)
    for ind in nonnegInds
        par[ind] = log(par[ind])
    end
    copyto!(par, p + l + 1 + (q * (q + 1)) >> 1, m.lγω)
    par[end] = log(m.lω[1])
    par
    #print("modelpar_to_optimpar par = ", par, "\n")
end

"""
    optimpar_to_modelpar!(m, par)
Translate optimization variables in `par` to the model parameters in `m`.
"""
function optimpar_to_modelpar!(
    m::varlmmModel, 
    par::Vector)
    p, q, l = size(m.data[1].X, 2), size(m.data[1].Z, 2), size(m.data[1].W, 2)
    copyto!(m.β, 1, par, 1, p)
    copyto!(m.τ, 1, par, p + 1, l)
    vec2ltri!(m.Lγ, @views par[(p + l + 1): p + l + (q * (q + 1)) >> 1])
    #exponentiate diagonals of m.Lγ to recover positivity constraint
    for i in 1:q
        m.Lγ[i, i] = exp(m.Lγ[i, i])
    end
    copyto!(m.lγω, 1, par, p + l + 1 + (q * (q + 1)) >> 1, q)
    m.lω[1] = exp(par[end])
    m
end

function MathProgBase.initialize(
    m::varlmmModel, 
    requested_features::Vector{Symbol}
    )
    for feat in requested_features
        if !(feat in [:Grad])
            error("Unsupported feature $feat")
        end
    end
end

MathProgBase.features_available(m::varlmmModel) = [:Grad]

function MathProgBase.eval_f(
    m::varlmmModel, 
    par::Vector)
    optimpar_to_modelpar!(m, par)
    MoMobjf!(m, false)
end

function MathProgBase.eval_grad_f(
    m::varlmmModel, 
    grad::Vector, 
    par::Vector)
    p, q, l = size(m.data[1].X, 2), size(m.data[1].Z, 2), size(m.data[1].W, 2)
    optimpar_to_modelpar!(m, par) 
    obj = MoMobjf!(m, true)
    # gradient wrt β
    copyto!(grad, m.∇β)
    # gradient wrt τ
    copyto!(grad, p + 1, m.∇τ)
    # gradient wrt Lγ
    copyto!(grad, p + l + 1, m.∇Lγ)
    nonnegInds = diagInds(q) .+ (p + l) #log(grad) chain rule term
    for i in eachindex(nonnegInds)
        grad[nonnegInds[i]] *= m.Lγ[i, i]
    end
    # gradient wrt lγω
    copyto!(grad, p + l + 1 + (q * (q + 1)) >> 1, m.∇lγω)
    # gradient wrt lω
    grad[end] = m.∇lω[1] * m.lω[1] #log(grad) chain rule term
    grad
end

function MathProgBase.eval_g(
    m::varlmmModel,
    g::Vector, 
    par::Vector)
    fill!(g, 0)
end
