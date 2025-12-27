function sampleβ!(params, y, X, priors)

    nsegments = length(params.intervals) - 1
    p = size(X,2)

    for k in 1:nsegments

        indk = (params.intervals[k]+1):params.intervals[k+1]

        yk = view(y, indk, :)
        Xk = view(X, indk, :)
        σₖ² = view(params.σ², k)

        Qpost = priors.β.Q + (Xk'*Xk) ./ σₖ²
        Qpostchol = cholesky(Symmetric(Qpost))

        μpost = Qpostchol \ (Xk'*yk) ./ σₖ²

        params.β[:,k] = μpost + Qpostchol.U \ randn(p)

    end   

end

function sampleσ²!(params, y, X, priors)

    nsegments = length(params.intervals) - 1

    for k in 1:nsegments

        indk = (params.intervals[k]+1):params.intervals[k+1]

        yk = view(y, indk, :)
        Xk = view(X, indk, :)
        βk = view(params.β, :, k)

        residk = yk - Xk*βk
        ssek = sum(residk.^2)

        postshape = length(indk)/2 + priors.σ².shape
        postscale = ssek/2 + priors.σ².scale

        postdist = InverseGamma(postshape, postscale)

        params.σ²[k] = rand(postdist)

    end   

end

bicumsum(x, y) = cumsum(x[1:(end-1)]) + reverse(cumsum(reverse(y[2:end])))

function samplec!(params, y, X)

    nbreaks = length(params.intervals) - 2

    for j in 1:nbreaks

        lb = params.intervals[j] + 1
        ub = params.intervals[j+2]

        indj = lb:ub

        yj = @view y[indj]
        Xj = @view X[indj,:]
        βleft = @view params.β[:,j]
        βright = @view params.β[:,j+1]
        σ²left = params.σ²[j]
        σ²right = params.σ²[j+1]

        loglikeleft = -0.5*( (yj - Xj*βleft).^2 / σ²left .+ log(σ²left) )
        loglikeright = -0.5*( (yj - Xj*βright).^2 / σ²right .+ log(σ²right) )

        logits = bicumsum(loglikeleft, loglikeright)

        probs = exp.(logits .- logsumexp(logits))

        params.intervals[j+1] = indj[rand(Categorical(probs))]

    end

end

function logdensity!(ld, params, y, X)

    nsegments = length(params.intervals) - 1

    for k in 1:nsegments

        indk = (params.intervals[k]+1):params.intervals[k+1]
        yk = view(y, indk, :)
        Xk = view(X, indk, :)
        βk = view(params.β, :, k)
        σₖ² = params.σ²[k]

        ld[indk] = -0.5*( (yk - Xk*βk).^2 ./ σₖ² .+ log(σₖ²) )

    end

end

function pwlr(y, X, w, priors, params, nsamps; getelpd::Bool = true)

    n, p = size(X)
    nsegments = length(params.intervals) - 1

    samples = (
        β = zeros(p, nsegments, nsamps),
        σ² = zeros(nsegments, nsamps),
        intervals = zeros(nsegments+1, nsamps)
    )

    winvsqrt = (1 ./ sqrt.(w))
    ys =  winvsqrt .* y
    Xs = winvsqrt .* X
    
    if getelpd
        ld = zeros(n, nsamps)
    end

    for i in 1:nsamps
        sampleβ!(params, ys, Xs, priors)
        sampleσ²!(params, ys, Xs, priors)
        samplec!(params, ys, Xs)

        samples.β[:,:,i] = params.β
        samples.σ²[:,i] = params.σ²
        samples.intervals[:,i] = params.intervals

        if getelpd
            ldi = view(ld, :, i)
            logdensity!(ldi, params, ys, Xs)
        end
    end

    elpd = zeros(n)
    if getelpd
        elpd += vec(logsumexp(ld, dims = 2)) .- log(nsamps)
        elpd -= vec(var(ld, dims = 2))
    end

    return samples, elpd

end

function simparams(X, priors, nbreaks; σcz = 1.0)

    n,p = size(X)
    nsegments = nbreaks + 1

    cz = σcz*randn(nsegments)
    cp = cumsum( exp.(cz[1:nbreaks]) / sum(exp.(cz)) )

    c = Int.(floor.(n*cp))

    intervals = [0; c; n]

    βdist = MvNormalCanon(priors.β.Q)
    σ²dist = InverseGamma(priors.σ².shape, priors.σ².scale)

    params = (β = zeros(p, nsegments), σ² = zeros(nsegments), intervals = intervals)

    for k in 1:nsegments

        βk = rand(βdist)
        σ²k = rand(σ²dist)

        params.β[:,k] = βk
        params.σ²[k] = σ²k

    end

    return params

end

function simdata(X, w, params)

    n = size(X,1)
    nsegments = length(params.intervals) - 1

    y = zeros(n)

    for k in 1:nsegments

        indk = (params.intervals[k]+1):params.intervals[k+1]
        y[indk] = @views X[indk,:]*params.β[:,k] + sqrt.(params.σ²[k] ./ w[indk]) .* randn(length(indk))

    end

    return y

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