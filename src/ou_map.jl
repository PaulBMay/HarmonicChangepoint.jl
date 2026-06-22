# =============================================================================
# MAP / ICM point estimate of the UK (OU-GP) changepoint model.
#
# Deterministic coordinate-ascent twin of `ou_gibbs_mv`: every random draw in the
# Gibbs sweep is replaced by its conditional MODE, so the sweep climbs the joint
# (log) posterior of (breaks, β, Σ, σ²g, G) instead of sampling it:
#   - breaks : argmax of the break-position split likelihood   (was rcat∘softmax)
#   - G      : RTS smoother MEAN                                (was ffbs_sample!)
#   - β      : Gaussian posterior MEAN / GLS                    (drop the draw)
#   - Σ_k    : inverse-Wishart MODE  Sc/(dfp+r+1)              (was rand IW)
#   - σ²g    : inverse-gamma MODE rate/(α+1), then CAPPED       (was rand IG)
#
# The σ²g CAP (not annealing) is what keeps the joint MAP well-posed: with σ²g
# bounded, the conditional mode of G is the penalised smoother mean and cannot
# interpolate the data, so abrupt steps are pushed onto breaks rather than
# absorbed by a runaway GP. Seeded with the same `init_breaks` binary
# segmentation. Iterates to convergence in the data log-likelihood.
#
# Returns S in the SAME NamedTuple layout as `ou_gibbs_mv` but with the sample
# axis = 1 (the mode), so `query_posterior` and the wall2wall extraction consume
# it unchanged. ICM is local (one break moved at a time); cap + binary-seg init
# stand in for the global optimal-partitioning DP, which is the later upgrade.
# =============================================================================

# ICM at a fixed number of breaks `nb`. Returns (S, loglik) where loglik is the
# data log-likelihood at the mode (used, penalised, for nb selection).
function ou_icm_mv(Y, t, w, X, ρ, nb, pr; σ2g_cap=0.1, gp=true,
                   maxit=100, tol=1e-6, minseg=4)
    n, p = size(X); r = size(Y,2); nseg = nb+1
    iv = init_breaks(Y, t, w, X, ρ, nb, 0.02, 0.02; minseg=minseg)
    β = zeros(p, r, nseg); G = zeros(n, r)
    σ2g = gp ? σ2g_cap : 0.0                      # start at the cap (max flexibility)
    Σ = zeros(r, r, nseg); for k in 1:nseg; Σ[:,:,k] = 0.01*Matrix(I,r,r); end

    # ---- preallocated workspace ----
    wsk  = OUWork(n)
    Uall = zeros(r, r, nseg); dall = zeros(r, nseg)
    resid = zeros(n, r); rot = zeros(n, r); Gt = zeros(n, r); wEbuf = zeros(n, r)
    Lfwd = zeros(n); Lbwd = zeros(n); bwdout = zeros(n); vbuf = zeros(n); gcol = zeros(n); wr = zeros(n)
    score = zeros(n)
    XtWX = zeros(p,p); wX = zeros(n,p); Prec = zeros(p,p); rhs = zeros(p); mβ = zeros(p); βt = zeros(p,r)
    Mbuf = zeros(r,r); dvec = zeros(r); Sc = zeros(r,r); tmpr = zeros(r); ev = zeros(r); Cbuf = zeros(r,r)

    prev_obj = -Inf; obj = -Inf
    @inbounds for it in 1:maxit
        # Σ eigendecomposition once per sweep
        for k in 1:nseg
            Uk = view(Uall,:,:,k); copyto!(Uk, view(Σ,:,:,k))
            dk, _ = LAPACK.syev!('V', 'U', Uk)
            for ℓ in 1:r; dall[ℓ,k] = max(dk[ℓ], 1e-10); end
        end

        # (a) breaks | β, Σ  — argmax of the split likelihood
        for j in 1:nb
            lb = Int(iv[j])+1; ub = Int(iv[j+2]); m = ub-lb+1
            m < 2minseg && continue
            ts = view(t, lb:ub)
            dL = view(dall,:,j); dR = view(dall,:,j+1)
            UL = view(Uall,:,:,j); UR = view(Uall,:,:,j+1)
            for c in 1:m; Lfwd[c] = 0.0; Lbwd[c] = 0.0; end
            mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,j))
            @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
            mul!(view(rot,1:m,:), view(resid,1:m,:), UL)
            for ℓ in 1:r
                dv = dL[ℓ]; for c in 1:m; vbuf[c] = dv/w[lb+c-1]; end
                kalman_forward!(wsk, view(rot,1:m,ℓ), ts, ρ, σ2g*dv, view(vbuf,1:m))
                for c in 1:m; Lfwd[c] += wsk.cumll[c]; end
            end
            mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,j+1))
            @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
            mul!(view(rot,1:m,:), view(resid,1:m,:), UR)
            for ℓ in 1:r
                dv = dR[ℓ]; for c in 1:m; vbuf[c] = dv/w[lb+c-1]; end
                seg_cumll_bwd!(bwdout, wsk, view(rot,1:m,ℓ), ts, ρ, σ2g*dv, view(vbuf,1:m))
                for c in 1:m; Lbwd[c] += bwdout[c]; end
            end
            lo = minseg; hi = m-minseg; L = hi-lo+1
            bestq = 1; bestv = -Inf
            for q in 1:L
                sv = Lfwd[lo+q-1] + Lbwd[lo+q]
                if sv > bestv; bestv = sv; bestq = q; end
            end
            iv[j+1] = lb + (lo + bestq - 1) - 1
        end

        # (b) G | β, Σ, σ2g  — RTS smoother MEAN
        if gp
            for k in 1:nseg
                lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
                ts = view(t, lb:ub); U = view(Uall,:,:,k); d = view(dall,:,k)
                mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,k))
                @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
                mul!(view(rot,1:m,:), view(resid,1:m,:), U)
                for ℓ in 1:r
                    dv = d[ℓ]; for c in 1:m; vbuf[c] = dv/w[lb+c-1]; end
                    kalman_smooth_mean!(view(gcol,1:m), wsk, view(rot,1:m,ℓ), ts, ρ, σ2g*dv, view(vbuf,1:m))
                    for c in 1:m; Gt[c,ℓ] = gcol[c]; end
                end
                mul!(view(G,lb:ub,:), view(Gt,1:m,:), U')
            end
        end

        # (c) β | G, Σ  — posterior MEAN
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
                for a in 1:p; βt[a,ℓ] = mβ[a]; end
            end
            mul!(view(β,:,:,k), βt, U')
        end

        # (d) σ2g | G, Σ  — inverse-gamma MODE, then capped
        if gp
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
            σ2g = min(rate / (pr.ag + r*n/2 + 1), σ2g_cap)
        end

        # (e) Σ_k | β, G, σ2g  — inverse-Wishart MODE  Sc/(dfp+r+1)
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
            Σ[:,:,k] .= Symmetric(Sc) ./ (dfp + r + 1)
        end

        # convergence: data log-likelihood at the current mode
        obj = 0.0
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
                obj += -0.5*(r*log(2π) + ldΣ - r*log(w[i]) + w[i]*q)
            end
        end
        (it > 1 && abs(obj - prev_obj) <= tol*(abs(obj)+1e-12)) && break
        prev_obj = obj
    end

    S = (β=reshape(copy(β),p,r,nseg,1), iv=reshape(copy(iv),nseg+1,1), σ2g=[σ2g],
         Σ=reshape(copy(Σ),r,r,nseg,1), G=reshape(copy(G),n,r,1))
    S, obj
end

# nb selection for the MAP fit by penalized data loglik. Unlike the sampler's
# δ·nb (calibrated to WAIC, which self-penalizes), MAP uses the MAXIMIZED data
# loglik, which always rises with more segments — so the default penalty is
# BIC-style: per added break, ½·k_seg·log(n), where k_seg = the free parameters
# an extra segment introduces (β: p·r, Σ: r(r+1)/2). Pass a numeric `δ` to force
# the flat δ·nb penalty instead. Mirrors `fit_adaptive_mv`'s greedy/exhaustive
# split. Returns (S, nb, loglik).
function fit_map_mv(Y,t,w,X,ρ,pr; maxnb=3, δ=nothing, σ2g_cap=0.1, gp=true,
                    maxit=100, tol=1e-6, greedy=false)
    n = size(Y,1); p = size(X,2); r = size(Y,2)
    pen = δ === nothing ? 0.5*(p*r + (r*(r+1))÷2)*log(n) : float(δ)   # per-break
    if greedy
        Sbest,llbest = ou_icm_mv(Y,t,w,X,ρ,0,pr; σ2g_cap=σ2g_cap, gp=gp, maxit=maxit, tol=tol)
        bestnb=0; bestscore=llbest; nb=0
        while nb < maxnb
            nb += 1
            S,ll = ou_icm_mv(Y,t,w,X,ρ,nb,pr; σ2g_cap=σ2g_cap, gp=gp, maxit=maxit, tol=tol)
            sc = ll - pen*nb
            sc > bestscore || break
            Sbest,llbest,bestnb,bestscore = S,ll,nb,sc
        end
        return Sbest, bestnb, llbest
    end
    fits=Vector{Any}(undef,maxnb+1); score=fill(-Inf,maxnb+1)
    for nb in 0:maxnb
        S,ll = ou_icm_mv(Y,t,w,X,ρ,nb,pr; σ2g_cap=σ2g_cap, gp=gp, maxit=maxit, tol=tol)
        fits[nb+1]=(S=S, e=ll); score[nb+1]=ll - pen*nb
    end
    b=argmax(score); fits[b].S, b-1, fits[b].e
end
