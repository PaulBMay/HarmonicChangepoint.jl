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
#
# The sampler is written for low allocation (cf. pwlr): every working array is
# preallocated once, the OU recursions run in-place through an OUWork, and each
# segment's Σ eigendecomposition is computed once per sweep and reused by the
# break / G / β / σ²g steps.
# =============================================================================

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

# In-place GP quadratic form M = Gᵀ R₀⁻¹ G (unit-var OU correlation) via the OU
# innovations; dvec is length-r scratch. G may be a view; indices are local.
function ou_quadmat!(M, dvec, G, t, ρ)
    r = size(G,2); fill!(M, 0.0)
    @inbounds for b in 1:r, c in 1:r
        M[b,c] += G[1,b]*G[1,c]
    end
    @inbounds for i in 2:size(G,1)
        φ = exp(-(t[i]-t[i-1])/ρ); iv = 1/(1-φ^2)
        for b in 1:r; dvec[b] = G[i,b] - φ*G[i-1,b]; end
        for b in 1:r, c in 1:r
            M[b,c] += dvec[b]*dvec[c]*iv
        end
    end
    M
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

    # ---- preallocated workspace (allocated once) ----
    ws   = OUWork(n)
    Uall = zeros(r, r, nseg); dall = zeros(r, nseg)          # per-segment Σ eigendecomp
    resid = zeros(n, r); rot = zeros(n, r); Gt = zeros(n, r); wEbuf = zeros(n, r)
    Lfwd = zeros(n); Lbwd = zeros(n); bwdout = zeros(n); vbuf = zeros(n); gcol = zeros(n); wr = zeros(n)
    score = zeros(n); probs = zeros(n)
    XtWX = zeros(p,p); wX = zeros(n,p); Prec = zeros(p,p); rhs = zeros(p); zvec = zeros(p); mβ = zeros(p); βt = zeros(p,r)
    Mbuf = zeros(r,r); dvec = zeros(r); Sc = zeros(r,r); tmpr = zeros(r); ev = zeros(r); Cbuf = zeros(r,r)

    @inbounds for it in 1:nsamps
        # Σ eigendecomposition once per sweep (Σ is constant until step (e)),
        # reused by the break / G / β / σ²g steps.
        for k in 1:nseg
            Uk = view(Uall,:,:,k); copyto!(Uk, view(Σ,:,:,k))
            dk, _ = LAPACK.syev!('V', 'U', Uk)
            for ℓ in 1:r; dall[ℓ,k] = max(dk[ℓ], 1e-10); end
        end

        # (a) breaks | β, Σ  (integrate G; per-band fwd/bwd Kalman summed over bands)
        for j in 1:nb
            lb = Int(iv[j])+1; ub = Int(iv[j+2]); m = ub-lb+1
            m < 2minseg && continue
            ts = view(t, lb:ub)
            dL = view(dall,:,j); dR = view(dall,:,j+1)
            UL = view(Uall,:,:,j); UR = view(Uall,:,:,j+1)
            for c in 1:m; Lfwd[c] = 0.0; Lbwd[c] = 0.0; end
            # left: rotate (Y-Xβ_j) by U_j, forward Kalman per band
            mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,j))
            @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
            mul!(view(rot,1:m,:), view(resid,1:m,:), UL)
            for ℓ in 1:r
                dv = dL[ℓ]; for c in 1:m; vbuf[c] = dv/w[lb+c-1]; end
                kalman_forward!(ws, view(rot,1:m,ℓ), ts, ρ, σ2g*dv, view(vbuf,1:m))
                for c in 1:m; Lfwd[c] += ws.cumll[c]; end
            end
            # right: rotate (Y-Xβ_{j+1}) by U_{j+1}, backward (suffix) Kalman per band
            mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,j+1))
            @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
            mul!(view(rot,1:m,:), view(resid,1:m,:), UR)
            for ℓ in 1:r
                dv = dR[ℓ]; for c in 1:m; vbuf[c] = dv/w[lb+c-1]; end
                seg_cumll_bwd!(bwdout, ws, view(rot,1:m,ℓ), ts, ρ, σ2g*dv, view(vbuf,1:m))
                for c in 1:m; Lbwd[c] += bwdout[c]; end
            end
            lo = minseg; hi = m-minseg; L = hi-lo+1
            for q in 1:L; score[q] = Lfwd[lo+q-1] + Lbwd[lo+q]; end
            softmax!(view(probs,1:L), view(score,1:L))
            iv[j+1] = lb + (lo + rcat(view(probs,1:L)) - 1) - 1
        end

        # (b) G | β, Σ, σ2g  (rotate by Σ eigvecs, FFBS each band, unrotate)
        if gp
            for k in 1:nseg
                lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
                ts = view(t, lb:ub); U = view(Uall,:,:,k); d = view(dall,:,k)
                mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,k))
                @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
                mul!(view(rot,1:m,:), view(resid,1:m,:), U)
                for ℓ in 1:r
                    dv = d[ℓ]; for c in 1:m; vbuf[c] = dv/w[lb+c-1]; end
                    ffbs_sample!(view(gcol,1:m), ws, view(rot,1:m,ℓ), ts, ρ, σ2g*dv, view(vbuf,1:m); rng=rng)
                    for c in 1:m; Gt[c,ℓ] = gcol[c]; end
                end
                mul!(view(G,lb:ub,:), view(Gt,1:m,:), U')
            end
        end

        # (c) β | G, Σ  (rotate residual Y-G, per band weighted Bayes reg, unrotate)
        for k in 1:nseg
            lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
            U = view(Uall,:,:,k); d = view(dall,:,k); Xk = view(X,lb:ub,:)
            @views resid[1:m,:] .= Y[lb:ub,:] .- G[lb:ub,:]
            mul!(view(rot,1:m,:), view(resid,1:m,:), U)
            for c in 1:m, a in 1:p; wX[c,a] = w[lb+c-1]*Xk[c,a]; end
            mul!(XtWX, Xk', view(wX,1:m,:))
            for ℓ in 1:r
                dv = d[ℓ]
                Prec .= pr.βQ .+ XtWX ./ dv
                C = cholesky!(Symmetric(Prec))
                for c in 1:m; wr[c] = w[lb+c-1]*rot[c,ℓ]; end
                mul!(rhs, Xk', view(wr,1:m)); rhs ./= dv
                copyto!(mβ, rhs); ldiv!(C, mβ)
                randn!(rng, zvec); ldiv!(C.U, zvec)
                for a in 1:p; βt[a,ℓ] = mβ[a] + zvec[a]; end
            end
            mul!(view(β,:,:,k), βt, U')
        end

        # (d) σ2g | G, Σ   (rate += 0.5 tr(Σ⁻¹ M_k) = 0.5 Σ_ℓ (u_ℓᵀ M u_ℓ)/d_ℓ)
        if gp && it > anneal_until
            rate = pr.bg
            for k in 1:nseg
                lb = Int(iv[k])+1; ub = Int(iv[k+1])
                ou_quadmat!(Mbuf, dvec, view(G,lb:ub,:), view(t,lb:ub), ρ)
                U = view(Uall,:,:,k); d = view(dall,:,k)
                for ℓ in 1:r
                    mul!(tmpr, Mbuf, view(U,:,ℓ))
                    rate += 0.5 * dot(view(U,:,ℓ), tmpr) / d[ℓ]
                end
            end
            σ2g = rand(rng, InverseGamma(pr.ag + r*n/2, rate))
        end

        # (e) Σ_k | β, G, σ2g   (combined IW: noise + GP, df += 2 n_k)
        for k in 1:nseg
            lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
            mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,k))
            @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:] .- G[lb:ub,:]
            for c in 1:m, b in 1:r; wEbuf[c,b] = w[lb+c-1]*resid[c,b]; end
            mul!(Sc, view(resid,1:m,:)', view(wEbuf,1:m,:))
            Sc .+= pr.ΣScale
            dfp = pr.Σdf + m
            if gp
                ou_quadmat!(Mbuf, dvec, view(G,lb:ub,:), view(t,lb:ub), ρ)
                Sc .+= Mbuf ./ σ2g; dfp += m
            end
            Σ[:,:,k] .= rand(rng, InverseWishart(dfp, Matrix(Symmetric(Sc))))
        end

        if it > nburn
            s = it - nburn; S.β[:,:,:,s] .= β; S.iv[:,s] .= iv; S.σ2g[s] = σ2g
            S.Σ[:,:,:,s] .= Σ; S.G[:,:,s] .= G
            for k in 1:nseg
                lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
                copyto!(Cbuf, view(Σ,:,:,k)); C = cholesky!(Symmetric(Cbuf))
                ldΣ = 0.0; for b in 1:r; ldΣ += 2*log(C.U[b,b]); end
                mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,k))
                @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:] .- G[lb:ub,:]
                for c in 1:m
                    i = lb+c-1
                    for b in 1:r; ev[b] = resid[c,b]; end
                    ldiv!(C.L, ev); q = dot(ev, ev)
                    ld[i,s] = -0.5*(r*log(2π) + ldΣ - r*log(w[i]) + w[i]*q)
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

# nb selection by penalized score elpd - δ·nb (δ = per-break complexity penalty).
# Returns (S, nb, elpd).
#   greedy=false (default): fit ALL nb=0..maxnb and pick argmax(score). A poor
#     intermediate nb (e.g. a single break that can't capture a disturbance+
#     recovery) can't abort the ladder, so an nb whose gain only appears beyond
#     a non-improving step is still reachable.
#   greedy=true: climb nb while the score keeps improving, stop at the first
#     non-improving step. Cheaper (skips higher nb once a break stops paying for
#     itself) and gives the SAME nb as non-greedy whenever score is unimodal in
#     nb; only differs on the rare non-monotone pixel (e.g. two separate abrupt
#     events where one mid break is worse than none but two help).
function fit_adaptive_mv(Y,t,w,X,ρ,pr,nsamps,nburn; maxnb=3, δ=4.0, rng=Random.default_rng(), gp=true, greedy=false)
    if greedy
        Sbest,elpdbest = ou_gibbs_mv(Y,t,w,X,ρ,0,pr,nsamps,nburn; rng=rng, gp=gp)
        bestnb=0; bestscore=sum(elpdbest)               # score at nb=0 (δ·0)
        nb=0
        while nb < maxnb
            nb += 1
            S,elpd = ou_gibbs_mv(Y,t,w,X,ρ,nb,pr,nsamps,nburn; rng=rng, gp=gp)
            sc = sum(elpd) - δ*nb
            sc > bestscore || break
            Sbest,elpdbest,bestnb,bestscore = S,elpd,nb,sc
        end
        return Sbest, bestnb, sum(elpdbest)
    end
    fits=Vector{Any}(undef,maxnb+1); score=fill(-Inf,maxnb+1)
    for nb in 0:maxnb
        S,elpd = ou_gibbs_mv(Y,t,w,X,ρ,nb,pr,nsamps,nburn; rng=rng, gp=gp)
        fits[nb+1]=(S=S, e=sum(elpd)); score[nb+1]=sum(elpd) - δ*nb
    end
    b=argmax(score); fits[b].S, b-1, fits[b].e
end
