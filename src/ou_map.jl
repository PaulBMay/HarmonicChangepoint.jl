# =============================================================================
# MAP point estimate of the UK (OU-GP) changepoint model — latent G MARGINALIZED,
# parameterized by the GP VARIANCE FRACTION gpfrac ∈ (0,1).
#
# Within a segment the residual splits into a smooth OU-GP trend G and white noise
# E that SHARE the cross-band covariance Σ; gpfrac is the trend's share of the
# (unit-weight) variance:
#     Var(G_ib) = gpfrac·Σ_bb,   Var(E_ib) = (1-gpfrac)·Σ_bb / w_i
# so Y-Xβ = √gpfrac·G̃ + √(1-gpfrac)·Ẽ (convex split of VARIANCES). This replaces
# the old unbounded σ²g (= gpfrac/(1-gpfrac)) with an interpretable, BOUNDED knob:
# gpfrac=0 → no trend, gpfrac=1 → pure GP. No cap is needed (the support bounds
# it) and no σ²g magnitude prior is needed — Σ's inverse-Wishart prior provides
# the soft regularization of the total variance, and the data self-regularizes
# gpfrac (→1 forces noise→0, which any white scatter penalizes; →0 is no-trend).
#
# Joint MAP over (gpfrac, G) would still be degenerate, so G is MARGINALIZED out
# of the objective; with shared-Σ each segment is matrix-normal
#     Y_k ~ MN(X_k β_k, A_k, Σ_k),  A_k = gpfrac·R₀(ρ) + (1-gpfrac)·diag(1/w)
# and the conditional modes are closed-form / 1-D:
#   - β_k    : GLS under A_k       β_k = (βQ + XᵀA⁻¹X)⁻¹ XᵀA⁻¹Y     (Σ-free)
#   - Σ_k    : inverse-Wishart mode (ΣScale + RΣ)/(Σdf + n_k + r + 1),
#              RΣ = (Y-Xβ)ᵀ A⁻¹ (Y-Xβ)   (Σ = TOTAL variance now)
#   - gpfrac : 1-D maximizer over (0,1) of the marginal posterior (+ Beta prior)
#   - breaks : argmax of the split likelihood (integrates G)
# All A⁻¹ work is the scalar Kalman filter (kalman_innov! → standardized
# innovations whose inner products give the XᵀA⁻¹· cross-products + log|A|). G is
# recovered post-hoc per band by kalman_smooth_mean! on the GLS residual.
#
# Returns S in the SAME layout as ou_gibbs_mv (sample axis = 1, the mode) PLUS a
# `gpfrac` field; the `σ2g` field carries gpfrac/(1-gpfrac) for direct comparison
# with the Gibbs. query_posterior / the wall2wall extraction consume it unchanged.
# `Σ` is the TOTAL variance (differs from the Gibbs' GP/noise col-cov by 1-gpfrac;
# nothing downstream consumes S.Σ). The β prior βQ is applied per band (exact for
# βQ ∝ I). gp=false (gpfrac≡0) reduces A_k to diag(1/w) — the matched WLS control.
# =============================================================================

# Golden-section maximizer of a unimodal f on [lo, hi].
function _golden_max(f, lo, hi; tol=1e-5, maxit=80)
    r = (sqrt(5)-1)/2; c = (3-sqrt(5))/2
    a = lo; b = hi; h = b-a
    x1 = a + c*h; x2 = a + r*h; f1 = f(x1); f2 = f(x2)
    for _ in 1:maxit
        if f1 > f2
            b, x2, f2 = x2, x1, f1; h = b-a; x1 = a + c*h; f1 = f(x1)
        else
            a, x1, f1 = x1, x2, f2; h = b-a; x2 = a + r*h; f2 = f(x2)
        end
        h < tol && break
    end
    f1 > f2 ? x1 : x2
end

# Marginalized ICM at a fixed number of breaks `nb`. Returns (S, loglik), loglik =
# the MARGINAL data log-likelihood at the mode (used, penalized, for nb selection).
# `aprop`,`bprop` parameterize an optional Beta(aprop,bprop) prior on gpfrac
# (default uniform → the Σ prior alone regularizes the variance magnitude).
function ou_icm_mv(Y, t, w, X, ρ, nb, pr; gp=true, maxit=100, tol=1e-6, minseg=4,
                   aprop=1.0, bprop=1.0, frac_floor=1e-4)
    n, p = size(X); r = size(Y,2); nseg = nb+1
    iv = init_breaks(Y, t, w, X, ρ, nb, 0.02, 0.02; minseg=minseg)
    β = zeros(p, r, nseg)
    gpfrac = gp ? 0.5 : 0.0
    Σ  = zeros(r, r, nseg); for k in 1:nseg; Σ[:,:,k] = 0.01*Matrix(I,r,r); end
    Σi = zeros(r, r, nseg); logdetΣ = zeros(nseg)
    vinv = similar(t); @inbounds for i in 1:n; vinv[i] = 1.0/w[i]; end
    fhi = 1.0 - frac_floor
    has_prior = (aprop != 1.0) || (bprop != 1.0)
    betalp(f) = has_prior ? (aprop-1)*log(f) + (bprop-1)*log(1-f) : 0.0

    # ---- workspace ----
    wsk  = OUWork(n)
    Uall = zeros(r, r, nseg); dall = zeros(r, nseg)
    resid = zeros(n, r); rot = zeros(n, r)
    Lfwd = zeros(n); Lbwd = zeros(n); bwdout = zeros(n); vbuf = zeros(n); vp = zeros(n)
    ZX = zeros(n, p); ZY = zeros(n, r); zcol = zeros(n)
    MXX = zeros(p,p); MXY = zeros(p,r); MYY = zeros(r,r); Prec = zeros(p,p)
    RΣ = zeros(r,r); Sc = zeros(r,r); tmp_pr = zeros(p,r); Cbuf = zeros(r,r)

    # marginal loglik (s-dependent part) of all segments at a given gpfrac f, with
    # β,Σ held at the current iterate (used by the 1-D gpfrac search + convergence).
    function marg_loglik(f; full=false)
        tot = betalp(f)
        for k in 1:nseg
            lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
            ts = view(t, lb:ub)
            for c in 1:m; vp[c] = (1-f)*vinv[lb+c-1]; end
            mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,k))
            @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
            ldA = 0.0
            for b in 1:r
                ldA = kalman_innov!(view(zcol,1:m), wsk, view(resid,1:m,b), ts, ρ, f, view(vp,1:m))
                for c in 1:m; ZY[c,b] = zcol[c]; end
            end
            mul!(RΣ, view(ZY,1:m,:)', view(ZY,1:m,:))
            tr = 0.0; Si = view(Σi,:,:,k)
            for b in 1:r, c in 1:r; tr += Si[b,c]*RΣ[c,b]; end
            tot += -0.5*(r*ldA + tr)
            full && (tot += -0.5*(m*logdetΣ[k] + m*r*log(2π)))
        end
        tot
    end

    prev_obj = -Inf; obj = -Inf
    @inbounds for it in 1:maxit
        # eigendecomposition of Σ (for the break step's rotation only)
        for k in 1:nseg
            Uk = view(Uall,:,:,k); copyto!(Uk, view(Σ,:,:,k))
            dk, _ = LAPACK.syev!('V', 'U', Uk)
            for ℓ in 1:r; dall[ℓ,k] = max(dk[ℓ], 1e-10); end
        end

        # (a) breaks | β, Σ, gpfrac — argmax of the split likelihood (G integrated)
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
                dv = dL[ℓ]; for c in 1:m; vbuf[c] = (1-gpfrac)*dv/w[lb+c-1]; end
                kalman_forward!(wsk, view(rot,1:m,ℓ), ts, ρ, gpfrac*dv, view(vbuf,1:m))
                for c in 1:m; Lfwd[c] += wsk.cumll[c]; end
            end
            mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,j+1))
            @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
            mul!(view(rot,1:m,:), view(resid,1:m,:), UR)
            for ℓ in 1:r
                dv = dR[ℓ]; for c in 1:m; vbuf[c] = (1-gpfrac)*dv/w[lb+c-1]; end
                seg_cumll_bwd!(bwdout, wsk, view(rot,1:m,ℓ), ts, ρ, gpfrac*dv, view(vbuf,1:m))
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

        # (b) β, Σ | gpfrac — GLS + IW mode under the marginal MN(Xβ, A, Σ)
        for k in 1:nseg
            lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
            ts = view(t, lb:ub)
            for c in 1:m; vp[c] = (1-gpfrac)*vinv[lb+c-1]; end
            ldA = 0.0
            for a in 1:p
                ldA = kalman_innov!(view(zcol,1:m), wsk, view(X,lb:ub,a), ts, ρ, gpfrac, view(vp,1:m))
                for c in 1:m; ZX[c,a] = zcol[c]; end
            end
            for b in 1:r
                kalman_innov!(view(zcol,1:m), wsk, view(Y,lb:ub,b), ts, ρ, gpfrac, view(vp,1:m))
                for c in 1:m; ZY[c,b] = zcol[c]; end
            end
            mul!(MXX, view(ZX,1:m,:)', view(ZX,1:m,:))      # XᵀA⁻¹X
            mul!(MXY, view(ZX,1:m,:)', view(ZY,1:m,:))      # XᵀA⁻¹Y
            mul!(MYY, view(ZY,1:m,:)', view(ZY,1:m,:))      # YᵀA⁻¹Y
            Prec .= pr.βQ .+ MXX
            Cβ = cholesky!(Symmetric(Prec))
            copyto!(view(β,:,:,k), MXY); ldiv!(Cβ, view(β,:,:,k))
            # RΣ = MYY - MXYᵀβ - βᵀMXY + βᵀMXX β
            mul!(tmp_pr, MXX, view(β,:,:,k))
            RΣ .= MYY
            mul!(RΣ, view(β,:,:,k)', MXY, -1.0, 1.0)
            mul!(RΣ, MXY', view(β,:,:,k), -1.0, 1.0)
            mul!(RΣ, view(β,:,:,k)', tmp_pr, 1.0, 1.0)
            Sc .= pr.ΣScale .+ Symmetric(RΣ)
            Σ[:,:,k] .= Sc ./ (pr.Σdf + m + r + 1)
            copyto!(Cbuf, view(Σ,:,:,k)); Cs = cholesky!(Symmetric(Cbuf))
            ld = 0.0; for b in 1:r; ld += 2*log(Cs.U[b,b]); end
            logdetΣ[k] = ld
            Si = view(Σi,:,:,k); copyto!(Si, Matrix(I,r,r)); ldiv!(Cs, Si)
        end

        # (d) gpfrac | β, Σ — 1-D maximize the marginal posterior on (0,1)
        if gp
            gpfrac = _golden_max(marg_loglik, frac_floor, fhi)
        end

        # convergence: full marginal data log-likelihood at the current mode
        obj = marg_loglik(gpfrac; full=true)
        (it > 1 && abs(obj - prev_obj) <= tol*(abs(obj)+1e-12)) && break
        prev_obj = obj
    end

    # ---- reconstruct G: per band, smoother mean gpfrac·R₀ A⁻¹ (Y-Xβ) ----
    G = zeros(n, r)
    if gp
        for k in 1:nseg
            lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
            ts = view(t, lb:ub)
            for c in 1:m; vp[c] = (1-gpfrac)*vinv[lb+c-1]; end
            mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,k))
            @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
            for b in 1:r
                kalman_smooth_mean!(view(zcol,1:m), wsk, view(resid,1:m,b), ts, ρ, gpfrac, view(vp,1:m))
                for c in 1:m; G[lb+c-1,b] = zcol[c]; end
            end
        end
    end

    σ2g_equiv = gpfrac >= 1.0 ? Inf : gpfrac/(1-gpfrac)
    S = (β=reshape(copy(β),p,r,nseg,1), iv=reshape(copy(iv),nseg+1,1),
         gpfrac=[gpfrac], σ2g=[σ2g_equiv],
         Σ=reshape(copy(Σ),r,r,nseg,1), G=reshape(G,n,r,1))
    S, obj
end

# nb selection for the MAP fit by penalized MARGINAL loglik. MAP maximises the
# (G-integrated) marginal likelihood, which still rises with nb (each break frees
# β: p·r and Σ: r(r+1)/2 params), so the default penalty is BIC-style:
# ½·k_seg·log(n) per break. Pass a numeric `δ` to force a flat δ·nb penalty.
# Mirrors `fit_adaptive_mv`'s greedy/exhaustive split. Returns (S, nb, loglik).
function fit_map_mv(Y,t,w,X,ρ,pr; maxnb=3, δ=nothing, gp=true, maxit=100, tol=1e-6,
                    greedy=false, aprop=1.0, bprop=1.0)
    n = size(Y,1); p = size(X,2); r = size(Y,2)
    pen = δ === nothing ? 0.5*(p*r + (r*(r+1))÷2)*log(n) : float(δ)   # per-break
    kw = (gp=gp, maxit=maxit, tol=tol, aprop=aprop, bprop=bprop)
    if greedy
        Sbest,llbest = ou_icm_mv(Y,t,w,X,ρ,0,pr; kw...)
        bestnb=0; bestscore=llbest; nb=0
        while nb < maxnb
            nb += 1
            S,ll = ou_icm_mv(Y,t,w,X,ρ,nb,pr; kw...)
            sc = ll - pen*nb
            sc > bestscore || break
            Sbest,llbest,bestnb,bestscore = S,ll,nb,sc
        end
        return Sbest, bestnb, llbest
    end
    fits=Vector{Any}(undef,maxnb+1); score=fill(-Inf,maxnb+1)
    for nb in 0:maxnb
        S,ll = ou_icm_mv(Y,t,w,X,ρ,nb,pr; kw...)
        fits[nb+1]=(S=S, e=ll); score[nb+1]=ll - pen*nb
    end
    b=argmax(score); fits[b].S, b-1, fits[b].e
end
