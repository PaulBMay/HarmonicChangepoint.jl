# =============================================================================
# Universal-kriging changepoint variant — multiband Gibbs sampler.
#
# Separate trend model from `pwlr`. Per segment:
#   Y_k = X_k β_k + G_k + E_k        (Y_k: n_k × r)
#   separable, SHARED-Σ:  vec(G_k) ~ N(0, σ²g R₀(ρ) ⊗ Σ_k),  E rows ~ N(0, Σ_k/w_i)
# Sharing Σ => fully conjugate (Σ: combined IW from noise + GP innovations,
# df += 2 n_k) and eigen-whitening of Σ decouples the r bands into independent
# scalar OU problems for both FFBS (sampling G) and the break sampler (forward/
# backward Kalman split-likelihoods, summed over bands). Range ρ is fixed;
# sampled: β_k, G, σ²g, Σ_k, breaks.
#
# X here carries only intercept + harmonics (drop the linear slope) — the OU
# latent G carries the within-segment trend. Public entry points:
#   - ou_gibbs_mv     : sample at a fixed number of breaks `nb`
#   - fit_adaptive_mv : fit nb = 0..maxnb and select by argmax(elpd - δ·nb)
# Set `gp=false` for a matched piecewise-linear-harmonic control (σ²g≡0, G≡0).
# =============================================================================

stdsoftmax(x) = (e = exp.(x .- maximum(x)); s = sum(e); e ./= s; e)
segranges(iv) = [(Int(iv[k])+1):Int(iv[k+1]) for k in 1:length(iv)-1]

# r×r GP quadratic-form matrix Gᵀ R₀⁻¹ G via OU innovations (unit-var correlation)
function ou_quadmat(G, t, ρ)
    r = size(G,2); M = (@view(G[1,:]))*(@view(G[1,:]))'
    @inbounds for i in 2:size(G,1)
        φ = exp(-(t[i]-t[i-1])/ρ); d = @view(G[i,:]) .- φ.*@view(G[i-1,:])
        M = M .+ (d*d')./(1-φ^2)
    end
    Symmetric(M)
end

# Greedy binary-segmentation init: seed breaks at the strongest split(s) using
# the Kalman split likelihood with a SMALL fixed σ2g (rigid GP) on OLS residuals.
function init_breaks(Y, t, w, X, ρ, nb, σ2g0, σ2ε0; minseg=4)
    n, r = size(Y)
    Res = Y .- X*((X'*(w.*X))\(X'*(w.*Y)))     # global OLS residual (de-mean/season)
    iv = [0, n]
    for _ in 1:nb
        bestgain=-Inf; bestpos=0; bestseg=0
        for s in 1:length(iv)-1
            lb=iv[s]+1; ub=iv[s+1]; m=ub-lb+1; m < 2minseg && continue
            idx=lb:ub; ts=@view t[idx]; ws=@view w[idx]
            base=0.0; Lfwd=zeros(m); Lbwd=zeros(m)
            for ℓ in 1:r
                col=@view Res[idx,ℓ]
                base += kalman_forward(col, ts, ρ, σ2g0, σ2ε0./ws).loglik
                Lfwd .+= seg_cumll_fwd(col, ts, ρ, σ2g0, σ2ε0./ws)
                Lbwd .+= seg_cumll_bwd(col, ts, ρ, σ2g0, σ2ε0./ws)
            end
            for c in minseg:m-minseg
                g = Lfwd[c]+Lbwd[c+1]-base
                if g>bestgain; bestgain=g; bestpos=lb+c-1; bestseg=s; end
            end
        end
        bestpos==0 && break
        insert!(iv, bestseg+1, bestpos); sort!(iv)
    end
    iv
end

# Multiband universal-kriging changepoint Gibbs at a fixed number of breaks `nb`.
# Returns (S, elpd): S holds the kept post-burn-in draws (β, iv, σ2g, Σ, G);
# elpd is the per-observation WAIC. Two stabilisers (see notes below):
#   - σ²g annealing (hold rigid for the first `anneal_frac` of burn-in) so breaks
#     LOCK onto steps before σ²g is freed for recovery, avoiding a bimodality
#     (flexible-GP-absorbs-step vs break+rigid-GP);
#   - profile-MAP break init via binary segmentation (`init_breaks`).
function ou_gibbs_mv(Y, t, w, X, ρ, nb, pr, nsamps, nburn; rng=Random.default_rng(),
                     minseg=4, anneal_frac=0.6, σ2g_clamp=0.05, gp=true)
    n, p = size(X); r = size(Y,2); nseg = nb+1
    iv = init_breaks(Y, t, w, X, ρ, nb, 0.02, 0.02; minseg=minseg)
    β = zeros(p, r, nseg); G = zeros(n, r); σ2g = gp ? σ2g_clamp : 0.0   # gp=false: matched linear control
    anneal_until = round(Int, anneal_frac*nburn)        # hold σ2g rigid to lock breaks
    Σ = zeros(r, r, nseg); for k in 1:nseg; Σ[:,:,k] = 0.01*Matrix(I,r,r); end
    keep = nsamps - nburn
    S = (β=zeros(p,r,nseg,keep), iv=zeros(nseg+1,keep), σ2g=zeros(keep),
         Σ=zeros(r,r,nseg,keep), G=zeros(n,r,keep))
    ld = zeros(n, keep)
    eig(k) = (F=eigen(Symmetric(Σ[:,:,k])); (U=F.vectors, d=max.(F.values,1e-10)))
    for it in 1:nsamps
        segs = segranges(iv)
        # (a) breaks | β, Σ  (integrate G; per-band fwd/bwd Kalman summed over bands)
        for j in 1:nb
            lb = Int(iv[j])+1; ub = Int(iv[j+2]); idx = lb:ub; m = length(idx)
            m < 2minseg && continue
            ts = @view t[idx]; ws = @view w[idx]
            eigj = eigen(Symmetric(Σ[:,:,j]));         dL = max.(eigj.values,1e-10); UL = eigj.vectors
            eigr = eigen(Symmetric(Σ[:,:,j+1]));       dR = max.(eigr.values,1e-10); UR = eigr.vectors
            RL = (@view(Y[idx,:]) .- @view(X[idx,:])*β[:,:,j])  * UL     # rotated left resid
            RR = (@view(Y[idx,:]) .- @view(X[idx,:])*β[:,:,j+1]) * UR    # rotated right resid
            Lfwd = zeros(m); Lbwd = zeros(m)
            for ℓ in 1:r
                Lfwd .+= seg_cumll_fwd(@view(RL[:,ℓ]), ts, ρ, σ2g*dL[ℓ], dL[ℓ]./ws)
                Lbwd .+= seg_cumll_bwd(@view(RR[:,ℓ]), ts, ρ, σ2g*dR[ℓ], dR[ℓ]./ws)
            end
            lo = minseg; hi = m-minseg
            score = [Lfwd[c] + Lbwd[c+1] for c in lo:hi]
            cl = (lo:hi)[rand(rng, Categorical(stdsoftmax(score)))]
            iv[j+1] = lb + cl - 1
        end
        segs = segranges(iv)
        # (b) G | β, Σ, σ2g  (rotate by Σ eigvecs, FFBS each band, unrotate)
        if gp
            for (k, idx) in enumerate(segs)
                U,d = eig(k); R = (@view(Y[idx,:]) .- @view(X[idx,:])*β[:,:,k]) * U
                Gt = similar(R)
                for ℓ in 1:r
                    Gt[:,ℓ] = ffbs_sample(@view(R[:,ℓ]), @view(t[idx]), ρ, σ2g*d[ℓ], d[ℓ]./@view(w[idx]); rng=rng)
                end
                G[idx,:] .= Gt * U'
            end
        end
        # (c) β | G, Σ  (rotate residual Y-G, per band weighted Bayes reg, unrotate)
        for (k, idx) in enumerate(segs)
            U,d = eig(k); Xk = @view X[idx,:]; wk = @view w[idx]
            R = (@view(Y[idx,:]) .- @view(G[idx,:])) * U
            XtWX = Xk'*(wk .* Xk); βt = zeros(p,r)
            for ℓ in 1:r
                Prec = Symmetric(pr.βQ .+ XtWX./d[ℓ]); C = cholesky(Prec)
                rhs = (Xk'*(wk .* @view(R[:,ℓ])))./d[ℓ]
                mβ = C \ rhs; z = randn(rng,p); ldiv!(C.U, z); βt[:,ℓ] = mβ .+ z
            end
            β[:,:,k] .= βt * U'
        end
        # (d) σ2g | G, Σ   (rate += 0.5 tr(Σ⁻¹ M_k)); clamped rigid during anneal
        if gp && it > anneal_until
            rate = pr.bg
            for (k, idx) in enumerate(segs)
                M = ou_quadmat(@view(G[idx,:]), @view(t[idx]), ρ); rate += 0.5*tr(Σ[:,:,k]\M)
            end
            σ2g = rand(rng, InverseGamma(pr.ag + r*n/2, rate))
        end
        # (e) Σ_k | β, G, σ2g   (combined IW: noise + GP, df += 2 n_k)
        for (k, idx) in enumerate(segs)
            E = @view(Y[idx,:]) .- @view(X[idx,:])*β[:,:,k] .- @view(G[idx,:]); wk = @view w[idx]
            Sc = E'*(wk .* E) .+ pr.ΣScale
            dfp = pr.Σdf + length(idx)
            if gp
                Sc = Sc .+ ou_quadmat(@view(G[idx,:]), @view(t[idx]), ρ)./σ2g; dfp += length(idx)
            end
            Σ[:,:,k] .= rand(rng, InverseWishart(dfp, Matrix(Symmetric(Sc))))
        end
        if it > nburn
            s = it - nburn; S.β[:,:,:,s] .= β; S.iv[:,s] .= iv; S.σ2g[s] = σ2g
            S.Σ[:,:,:,s] .= Σ; S.G[:,:,s] .= G
            for (k, idx) in enumerate(segs)
                Σi = inv(Symmetric(Σ[:,:,k])); ld5 = logdet(Symmetric(Σ[:,:,k]))
                E = @view(Y[idx,:]) .- @view(X[idx,:])*β[:,:,k] .- @view(G[idx,:])
                for (jj,i) in enumerate(idx)
                    e = @view E[jj,:]
                    ld[i,s] = -0.5*(r*log(2π) + ld5 - r*log(w[i]) + w[i]*dot(e, Σi, e))
                end
            end
        end
    end
    elpd = waic(ld)
    S, elpd
end

function waic(ld)
    S = size(ld,2); m = maximum(ld, dims=2)
    lse = vec(m) .+ log.(vec(sum(exp.(ld .- m), dims=2)))
    lse .- log(S) .- vec(var(ld, dims=2))
end

# NON-greedy nb selection: fit nb=0..maxnb, pick argmax(elpd - δ·nb). A poor
# intermediate nb (e.g. a single break that can't capture a disturbance+recovery)
# can't abort the ladder. δ is a per-break complexity penalty. Returns (S, nb, elpd).
function fit_adaptive_mv(Y,t,w,X,ρ,pr,nsamps,nburn; maxnb=3, δ=4.0, rng=Random.default_rng(), gp=true)
    fits=Vector{Any}(undef,maxnb+1); score=fill(-Inf,maxnb+1)
    for nb in 0:maxnb
        S,elpd = ou_gibbs_mv(Y,t,w,X,ρ,nb,pr,nsamps,nburn; rng=rng, gp=gp)
        fits[nb+1]=(S=S, e=sum(elpd)); score[nb+1]=sum(elpd) - δ*nb
    end
    b=argmax(score); fits[b].S, b-1, fits[b].e
end
