"""
    sandwich!(m::WSVarLmmModel)

Calculate the sandwich estimator of the asymptotic covariance of the parameters, 
based on values `m.Hββ`, `m.Hττ`, `m.HτLγ`, `m.HLγLγ`, `m.data[i].∇β`,
`m.data[i].∇τ`, and `m.data[i].∇Lγ`. `m.vcov` is updated by the sandwich 
estimator and returned.
"""
function sandwich!(m::WSVarLmmModel{T}) where T <: BlasReal
    p, q, l = m.p, m.q, m.l
    minv    = inv(m.m)
    # form A matrix in the sandwich formula
    fill!(m.Ainv, 0)
    m.Ainv[          1:p,                 1:p      ] = m.Hββ
    m.Ainv[    (p + 1):(p + l),     (p + 1):(p + l)] = m.Hττ
    m.Ainv[    (p + 1):(p + l), (p + l + 1):end    ] = m.HτLγ
    m.Ainv[(p + l + 1):end,     (p + l + 1):end    ] = m.HLγLγ
    copytri!(m.Ainv, 'U')
    lmul!(minv, m.Ainv)
    # form B matrix in the sandwich formula
    fill!(m.B, 0)
    for obs in m.data
        copyto!(m.ψ, 1    , obs.∇β)
        copyto!(m.ψ, p + 1, obs.∇τ)
        offset = p + l + 1
        @inbounds for j in 1:q, i in j:q
            m.ψ[offset] = obs.∇Lγ[i, j]
            offset += 1
        end
        BLAS.syr!('U', T(1), m.ψ, m.B)
    end
    copytri!(m.B, 'U')
    lmul!(minv, m.B)
    # calculuate A (pseudo)-inverse
    Aeval, Aevec = eigen(Symmetric(m.Ainv))
    @inbounds for j in 1:size(m.Ainv, 2)
        ej = Aeval[j]
        invsqrtej = ej > sqrt(eps(T)) ? inv(sqrt(ej)) : T(0)
        for i in 1:size(m.Ainv, 1)
            Aevec[i, j] *= invsqrtej
        end
    end
    mul!(m.Ainv, Aevec, transpose(Aevec))
    # calculate vcov
    mul!(Aevec , m.Ainv, m.B   ) # use Avec as scratch space
    mul!(m.vcov,  Aevec, m.Ainv)
    m.vcov .*= minv
end
