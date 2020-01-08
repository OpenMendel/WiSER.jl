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

#gets indicies for diagonal elements of matrix's vech elements (ensure >= 0)
function diagInds(n::Int)
    inds = [1]
    for i in 1:(n - 1)
        ind = inds[i] + (n - i + 1)
        push!(inds, ind)
    end
    inds
end

#takes vech object and transforms it to lower triangular matrix
function vec2ltri(v::AbstractVector{T}, z::T = zero(T)) where {T <: Real}
    n = length(v)
    s = round(Int, (sqrt(8n + 1) - 1) / 2)
    s * (s + 1) / 2 == n || error("vec2utri: length of vector is not triangular")
    [i >= j ? v[round(Int, j * (j - 1) / 2 + i)] : z for i=1:s, j=1:s]
end

"""
update_res!(obs, β)
Update the residual vector according to `β`.
"""
function update_res!(
    obs::varlmmObs{T}, 
    β::Vector{T}
    ) where T <: BlasReal
    copyto!(obs.res, obs.y)
    BLAS.gemv!('N', -one(T), obs.X, β, one(T), obs.res) # obs.res - obs.X * β = obs.y - obs.X * β
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

function exp!(x::Array{T}) where T <: BlasReal
    copyto!(x, exp.(x))
end

"""
MoMobjf(obs::varlmmObs{T},
β::Vector{T},
τ::T, # inverse of linear regression variance
Lγ::Matrix{Float64},
lγω::Vector{Float64},
lω::Float64,
needgrad::Bool = true
) where T <: BlasReal
Evaluates the method of moments objective function for the given data and parameter estimates
    and also the gradient.
"""
function MoMobjf!(obs::varlmmObs{T},
        β::Vector{T},
        τ::Vector{T},
        Lγ::Matrix{T},
        lγω::Vector{T},
        lω::Vector{T},
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
        eV = exp(0.5 * (lγω' * lγω + lω[1]^2 + 2 * lγω' * Lγ * lγω + (lγω' * Lγ) * (Lγ' * lγω)))
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
        objvalue = sum(abs2, obs.storage_nn) # Is this most efficient way to get? 

        #gradient
        if needgrad
            # wrt β
            mul!(obs.storage_n1, obs.storage_nn, obs.res)
            mul!(obs.∇β, transpose(obs.X), obs.storage_n1)
            #BLAS.gemv!('N', -inv(1 + qf), obs.xtz, obs.storage_q2, one(T), obs.∇β)
            obs.∇β .*= -2 #necessary? it's proportional to this 

            # wrt τ
            mul!(obs.storage_n1, Diagonal(obs.storage_nn), obs.Wτ)
            eVsumRiW = -eV * sum(obs.storage_n1) #need for gradient wrt Lγ
            mul!(obs.∇τ, transpose(obs.W), obs.storage_n1)
            obs.∇τ .*= -eV 

            # wrt Lγ
            # overwrite storage_qn 
            mul!(obs.storage_qn, transpose(obs.Z), obs.storage_nn)
            mul!(obs.storage_qq, obs.storage_qn, obs.Z)
            mul!(obs.storage_qq2, obs.storage_qq, Lγ)
            BLAS.syrk!('L', 'N', eVsumRiW, lγω, 0.0, obs.storage_qq) 
            copytri!(obs.storage_qq, 'L')
            BLAS.gemm!('N', 'N', 1.0, obs.storage_qq, I + Lγ, -2.0, obs.storage_qq2) 
            #obs.∇Lγ = vech(obs.storage_qq) #immutable
            copyto!(obs.∇Lγ, vech(obs.storage_qq))

            # wrt lγω
            obs.∇lγω .= eVsumRiW * (Lγ * lγω  + transpose(Lγ) * lγω + Lγ * transpose(Lγ) * lγω + lγω) 
            #BLAS.axpy!(eVsumRiW, Lγ * lγω, obs.∇lγω)
            #BLAS.axpy!(eVsumRiW, transpose(Lγ) * lγω, obs.∇lγω)
            #BLAS.axpy!(eVsumRiW, Lγ * transpose(Lγ) * lγω, obs.∇lγω)
            #BLAS.axpy!(eVsumRiW, lγω, obs.∇lγω)

            # wrt lω
            #obs.∇lω = eVsumRiW * lω #immutable
            mul!(obs.∇lω, [eVsumRiW], lω)
            
        end
    return objvalue
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
    for i in eachindex(m.data)
        objvalue += MoMobjf!(m.data[i], m.β, m.τ, m.Lγ, m.lγω, m.lω, needgrad)
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
    npar = Int(p + l + (q + 1) * (q + 2) / 2)
    # since X includes a column of 1, p is the number of mean parameters
    # the cholesky factor for the q+1xq+1 random effect mx has ((q + 1) * (q + 2))/2 values

    optm = MathProgBase.NonlinearModel(solver)
    lb = fill(-Inf, npar) # error variance should be nonnegative, will fix later
    ub = fill(Inf, npar)
    # diag elements of Lγ and lω must be > 0
    nonneginds = diagInds(q) .+ (p + l) #get diag inds of Lγ
    lb[nonneginds] .= 0
    lb[end] = 0

    MathProgBase.loadproblem!(optm, npar, 0, lb, ub, Float64[], Float64[], :Min, m)
    # starting point
    par0 = zeros(npar)
    modelpar_to_optimpar!(par0, m)
    #MathProgBase.eval_grad_f(m, grad, par0)
    MathProgBase.setwarmstart!(optm, par0)
    #print("after setwarmstart, par0 = ", par0, "\n")
    # optimize
    MathProgBase.optimize!(optm)
    # print("after optimize!, getsolution(optm) = ", MathProgBase.getsolution(optm), "\n")
    optstat = MathProgBase.status(optm)
    optstat == :Optimal || @warn("Optimization unsuccesful; got $optstat")
    # refresh gradient and Hessian
    optimpar_to_modelpar!(m, MathProgBase.getsolution(optm))
    # print("after optimize!, after optimpar_to_modelpar!, m.β = ", m.β, "\n")
    # print("after optimize!, after optimpar_to_modelpar!, m.Σ = ", m.Σ, "\n")
    # print("after optimize!, after optimpar_to_modelpar!, m.τ = ", m.τ, "\n")
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
    #print("modelpar_to_optimpar m.β = ", m.β, "\n")
    copyto!(par, m.β)
    copyto!(par, p + 1, m.τ) # take log and then exp() later to make the problem unconstrained
    #print("modelpar_to_optimpar m.β = ", m.Σ, "\n")
    #Σchol = cholesky(Symmetric(m.Σ))
    #print("modelpar_to_optimpar Σchol = ", Σchol, "\n")
    #m.ΣL .= Σchol.L
    # ?? what is the benefit of cholesky here?
    offset = p + 2
    copyto!(par, p + l + 1, vech(m.Lγ))
    copyto!(par, Int(p + l + q * (q + 1) / 2 + 1), m.lγω)
    par[end] = m.lω[1]
    #for j in 1:q
    #    par[offset] = log(m.ΣL[j, j]) # only the diagonal is constrained to be nonnegative
    #    offset += 1
    #    for i in j+1:q
    #        par[offset] = m.ΣL[i, j]
    #        offset += 1
    #    end
    #end
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
    #copyto!(m.Lγ, m.lγω, m.lω)
    m.Lγ .= vec2ltri(par[(p + l + 1): p + l + Int((q * (q + 1) / 2))])
    copyto!(m.lγω, 1, par, Int(p + l + q * (q + 1) / 2 + 1), q)
    m.lω[1] = par[end]
    #print("optimpar_to_modelpar par = ", par, "\n")
    # copyto!(dest, do, src, so, N)
    # Copy N elements from collection src starting at offset so, 
    # to array dest starting at offset do. Return dest.
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
    # gradient wrt lγω
    copyto!(grad, Int(p + l + q * (q + 1) / 2 + 1), m.∇lγω)
    # gradient wrt lω
    grad[end] = m.∇lω[1]
    obj
end
