function sampleβ!(params, buffer, Y, X, priors)
    
    # dimensions
    nsegments = length(params.intervals) - 1
    p = size(X,2)
    r = size(Y,2)

    # Allocation buffers
    XtX = buffer.XtX
    XtY = buffer.XtY
    XtYU = buffer.XtYU
    Qp = buffer.Qp
    U = buffer.U
    d = buffer.d
    βtilde = buffer.βtilde
    μp = buffer.μp
    z = buffer.z

    # Loop through piecewice segments
    @views for k in 1:nsegments

        # Subset relevant quants
        indk = (params.intervals[k]+1):params.intervals[k+1]
        Yk = Y[indk,:]
        Xk = X[indk,:]
        Σk = params.Σ[:,:,k]
        Uk = U[:,:,k]
        dk = d[:,k]

        # Compute eigenvalue decomp
        copyto!(Uk, Σk)
        dktemp, _ = LAPACK.syev!('V', 'U', Uk)
        dk .= dktemp

        # X^TX, X^TY, X^TYU
        mul!(XtX, Xk', Xk)
        mul!(XtY, Xk', Yk)
        mul!(XtYU, XtY, Uk)

        # Loop through responses
        for ℓ in 1:r
            # Posterior precision
            copyto!(Qp, priors.β.Q)
            @. Qp += (1/dk[ℓ])*XtX
            Qpc = cholesky!(Qp)
            # Posterior mean
            ldiv!(μp, Qpc, XtYU[:,ℓ])
            μp ./= dk[ℓ]
            # Random variation
            z .= randn(p)
            ldiv!(Qpc.U, z)
            # Write result
            @. βtilde[:,ℓ] = μp + z
        end

        # Unwhiten
        mul!(params.β[:,:,k], βtilde, Uk')

    end   

end

function sampleΣ!(params, buffer, Y, X, priors)
    # Dimensions
    nsegments = length(params.intervals) - 1
    r = size(Y,2)
    # Allocation buffers
    Scalep = buffer.Scalep
    resid = buffer.resid
    # Loop through segments
    @views for k in 1:nsegments
        # Slice relevant segment
        indk = (params.intervals[k]+1):params.intervals[k+1]
        Yk = Y[indk,:]
        Xk = X[indk,:]
        residk = resid[indk,:]
        βk = params.β[:,:,k]
        # Posterior scale and df for IW
        residk .= Yk
        mul!(residk, Xk, βk, -1.0, 1.0)  # residk .= -Xk*βk
        mul!(Scalep, residk', residk)
        Scalep .+= priors.Σ.Scale
        dfp = length(indk) + priors.Σ.df 
        # Sample
        postdist = InverseWishart(dfp, Scalep)
        params.Σ[:,:,k] .= rand(postdist)
    end   
end

function bicumsum!(out, x, y)
    n = length(x)
    @assert length(out) >= n - 1
    fill!(out, 0.0)
    leftsum = 0.0
    rightsum = 0.0
    @inbounds for i in 1:(n-1)
        leftsum += x[i]
        out[i] += leftsum
    end
    @inbounds for i in n:(-1):2
        rightsum += y[i]
        out[i-1] += rightsum
    end
end

function rcat(probs::AbstractVector)
    u = rand()                  # uniform [0,1)
    csum = 0.0
    for (i, p) in enumerate(probs)
        csum += p
        if u <= csum
            return i
        end
    end
    return length(probs)         # safety in case of rounding errors
end

function samplec!(params, buffer, Y, X)
    # Dimensions
    nbreaks = length(params.intervals) - 2
    r = size(Y,2)
    # Allocation buffers
    U = buffer.U
    d = buffer.d
    resid = buffer.resid
    residU = buffer.residU
    lll = buffer.lll
    llr = buffer.llr
    logits = buffer.logits
    probs = buffer.probs
    # Loop through breaks
    @views for j in 1:nbreaks
        # Subset relevant quants
        lb = params.intervals[j] + 1
        ub = params.intervals[j+2]
        indj = lb:ub

        Yj = Y[indj,:]
        Xj = X[indj,:]
        residj = resid[indj,:]
        residUj = residU[indj,:]
        lllj = lll[indj]
        llrj = llr[indj]
        logitsj = logits[lb:(ub-1)]
        probsj = probs[lb:(ub-1)]
        βleft = params.β[:,:,j]
        βright = params.β[:,:,j+1]
        Uleft = U[:,:,j]
        Uright = U[:,:,j+1]
        dleft = d[:,j]
        dright = d[:,j+1]
        
        # Left log-likes
        mul!(residj, Xj, βleft, -1.0, 0.0) 
        residj .+= Yj
        mul!(residUj, residj, Uleft)
        residUj .^= 2
        for ℓ in 1:r
            @. residUj[:,ℓ] /= dleft[ℓ]
        end
        lllj .=  vec(sum(residUj, dims = 2)) .+ sum(log.(dleft))
        lllj .*= -0.5
        # Right log-likes
        mul!(residj, Xj, βright, -1.0, 0.0) 
        residj .+= Yj
        mul!(residUj, residj, Uright)
        residUj .^= 2
        for ℓ in 1:r
            @. residUj[:,ℓ] /= dright[ℓ]
        end
        llrj .=  vec(sum(residUj, dims = 2)) .+ sum(log.(dright))
        llrj .*= -0.5

        bicumsum!(logitsj, lllj, llrj)
        softmax!(probsj, logitsj)
        params.intervals[j+1] = indj[rcat(probsj)]

    end

end

function logdensity!(ld, params, buffer, Y, X)

    nsegments = length(params.intervals) - 1
    r = size(Y,2)

    # Allocation buffers
    U = buffer.U
    d = buffer.d
    resid = buffer.resid
    residU = buffer.residU

    @views for k in 1:nsegments

        indk = (params.intervals[k]+1):params.intervals[k+1]
        Yk = Y[indk,:]
        Xk = X[indk,:]
        residk = resid[indk,:]
        residUk = residU[indk,:]
        βk = params.β[:,:,k]
        Uk = U[:,:,k]
        dk = d[:,k]

        mul!(residk, Xk, βk, -1.0, 0.0) 
        residk .+= Yk
        mul!(residUk, residk, Uk)
        residUk .^= 2
        for ℓ in 1:r
            @. residUk[:,ℓ] /= dk[ℓ]
        end
        ld[indk] .=  -0.5*(vec(sum(residUk, dims = 2)) .+ sum(log.(dk)) )

    end

end

function pwlr(Y::AbstractMatrix, X, w, priors, params, nsamps; getelpd::Bool = true)

    n, p = size(X)
    r = size(Y,2)
    nsegments = length(params.intervals) - 1

    samples = (
        β = zeros(p, r, nsegments, nsamps),
        Σ = zeros(r, r, nsegments, nsamps),
        intervals = zeros(nsegments+1, nsamps)
    )

    buffer = (
        XtX = zeros(p,p),
        XtY = zeros(p,r),
        XtYU = zeros(p,r),
        Qp = zeros(p,p),
        U = zeros(r,r,nsegments),
        d = zeros(r,nsegments),
        βtilde = zeros(p,r),
        μp = zeros(p),
        z = zeros(p),
        Scalep = zeros(r,r),
        resid = zeros(n,r),
        residU = zeros(n,r),
        lll = zeros(n),
        llr = zeros(n),
        logits = zeros(n),
        probs = zeros(n)
    )

    Ys =  sqrt.(w) .* Y
    Xs = sqrt.(w) .* X
    
    if getelpd
        ld = zeros(n, nsamps)
    end

    for i in 1:nsamps
        sampleΣ!(params, buffer, Ys, Xs, priors)
        sampleβ!(params, buffer, Ys, Xs, priors)
        samplec!(params, buffer, Ys, Xs)

        samples.β[:,:,:,i] = params.β
        samples.Σ[:,:,:,i] = params.Σ
        samples.intervals[:,i] = params.intervals

        if getelpd
            ldi = view(ld, :, i)
            logdensity!(ldi, params, buffer, Ys, Xs)
        end
    end

    elpd = zeros(n)
    if getelpd
        elpd .+= vec(logsumexp(ld, dims = 2)) .- log(nsamps)
        elpd .-= vec(var(ld, dims = 2))
    end

    return samples, elpd

end

function simparams(X, priors, nbreaks; σcz = 1.0)

    n, p = size(X)
    r = size(priors.Σ.Scale,1)
    nsegments = nbreaks + 1

    cz = σcz*randn(nsegments)
    cp = cumsum( exp.(cz[1:nbreaks]) / sum(exp.(cz)) )

    c = Int.(floor.(n*cp))

    intervals = [0; c; n]

    βdist = MvNormalCanon(priors.β.Q)
    Σdist = InverseWishart(priors.Σ.df, priors.Σ.Scale)

    params = (β = zeros(p, r, nsegments), Σ = zeros(r,r,nsegments), intervals = intervals)

    for k in 1:nsegments

        βk = rand(βdist, r)
        Σk = rand(Σdist)

        params.β[:,:,k] .= βk
        params.Σ[:,:,k] .= Σk

    end

    return params

end

function simdata(X, w, params)

    n = size(X,1)
    r = size(params.Σ,1)
    nsegments = length(params.intervals) - 1

    Y = zeros(n,r)

    @views for k in 1:nsegments
        indk = (params.intervals[k]+1):params.intervals[k+1]
        error_dist = MvNormal(params.Σ[:,:,k])
        error = sqrt.(1 ./ w[indk]) .* rand(error_dist, length(indk))'
        Y[indk,:] = X[indk,:]*params.β[:,:,k] + error
    end

    return Y

end

##################

function harmonicdesign(time::AbstractVector, period::Number, nfreqs::Int; intercept = true, slope = true)

    n = length(time)
    p = 1*intercept + 1*slope + 2*nfreqs

    X = zeros(n, p)

    if intercept
        X[:,1] .= 1
    end
    if slope
        X[:,1+1*intercept] = time
    end

    harmonic_start = 1*intercept + 1*slope + 1

    for freq in 1:nfreqs
        ind = harmonic_start + 2*(freq - 1)
        X[:,ind] = @. sinpi(2*time/period)
        X[:,ind+1] = @. cospi(2*time/period)
    end

    return X

end