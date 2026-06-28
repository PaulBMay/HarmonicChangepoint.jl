# =============================================================================
# MAP point estimate of the UK (OU-GP) changepoint model — latent G MARGINALIZED,
# parameterized by the GP variance σ²g in the BASE-COVARIANCE parameterization
# (identical to ou_gibbs_mv's model).
#
# Within a segment the residual splits into a smooth OU-GP trend G and white noise
# E that SHARE the cross-band base covariance Σ:
#     Var(G_ib) = σ²g·Σ_bb,   Var(E_ib) = Σ_bb / w_i
# i.e. σ²g is the GP variance as a MULTIPLE of the i.i.d. (unit-weight) noise.
# With G marginalized, each segment is matrix-normal
#     Y_k ~ MN(X_k β_k, A_k, Σ_k),   A_k = σ²g·R₀(ρ) + diag(1/w)
# (Σ = the BASE/noise covariance — NOT the total), and the conditional modes are
# closed-form:
#   - β_k    : GLS under A_k       β_k = (βQ + XᵀA⁻¹X)⁻¹ XᵀA⁻¹Y     (Σ-free)
#   - Σ_k    : inverse-Wishart mode (ΣScale + RΣ)/(Σdf + n_k + r + 1),
#              RΣ = (Y-Xβ)ᵀ A⁻¹ (Y-Xβ)
#   - σ²g    : a single global scalar — found by a ROBUST 1-D search (coarse
#              log-grid bracket + golden refine) over the PROFILE marginal
#              posterior, i.e. with β,Σ RE-FIT (GLS/IW) at every candidate σ²g,
#              plus the IG(ag,bg) prior on σ²g (same as the Gibbs).
#   - breaks : argmax of the split likelihood (integrates G).
#
# This supersedes the earlier `gpfrac ∈ (0,1)` total-variance reparameterization,
# which had TWO defects that both inflated the estimate toward near-interpolation:
#   (1) the IW prior on the TOTAL variance: under A = f·R₀ + (1-f)·diag(1/w) the
#       prior's effective strength rides the (1-f) split, penalizing high-noise
#       (low-f) solutions more and manufacturing a slope toward f→1; and
#   (2) coordinate ascent (golden-search f with β,Σ FROZEN) mis-converged to a
#       high-f local mode even when the true profile peaked low.
# Putting the prior on the base covariance fixes (1); the re-fit 1-D profile
# search fixes (2). Both were masked on mid-range pixels (σ²g≈1) and only showed
# up where the data wants a near-flat GP (tiny σ²g).
#
# All A⁻¹ work is the scalar Kalman filter (kalman_innov! → standardized
# innovations whose inner products give the XᵀA⁻¹· cross-products + log|A|). G is
# recovered post-hoc per band by kalman_smooth_mean! on the GLS residual (the base
# covariance Σ_bb cancels in the smoother gain, so the per-band smoother with
# (σ²g, 1/w) gives the exact posterior mean).
#
# Returns S in the SAME layout as ou_gibbs_mv (sample axis = 1, the mode) PLUS a
# `gpfrac` field carrying σ²g/(1+σ²g) (the bounded GP variance FRACTION) for an
# interpretable readout. query_posterior / the wall2wall extraction consume it
# unchanged. gp=false (σ²g≡0) reduces A_k to diag(1/w) — the matched WLS control.
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
# the profile marginal log-posterior at the mode (used, penalized, for nb
# selection). σ²g carries an IG(pr.ag, pr.bg) prior (as in the Gibbs).
function ou_icm_mv(Y, t, w, X, ρ, nb, pr; gp=true, maxit=100, tol=1e-6, minseg=4,
                   s2g_lo=1e-4, s2g_hi=1e2, ngrid=25)
    n, p = size(X); r = size(Y,2); nseg = nb+1
    iv = init_breaks(Y, t, w, X, ρ, nb, 0.02, 0.02; minseg=minseg)
    β = zeros(p, r, nseg)
    s2g = gp ? 0.1 : 0.0
    Σ  = zeros(r, r, nseg); for k in 1:nseg; Σ[:,:,k] = 0.01*Matrix(I,r,r); end
    Σi = zeros(r, r, nseg); logdetΣ = zeros(nseg)
    vinv = similar(t); @inbounds for i in 1:n; vinv[i] = 1.0/w[i]; end
    has_ig = (pr.ag != 0.0) || (pr.bg != 0.0)
    iglp(s) = (gp && has_ig) ? (-(pr.ag+1)*log(s) - pr.bg/s) : 0.0

    # ---- workspace ----
    wsk  = OUWork(n)
    Uall = zeros(r, r, nseg); dall = zeros(r, nseg)
    resid = zeros(n, r); rot = zeros(n, r)
    Lfwd = zeros(n); Lbwd = zeros(n); bwdout = zeros(n); vbuf = zeros(n)
    ZX = zeros(n, p); ZY = zeros(n, r); zcol = zeros(n)
    MXX = zeros(p,p); MXY = zeros(p,r); MYY = zeros(r,r); Prec = zeros(p,p)
    RΣ = zeros(r,r); Sc = zeros(r,r); tmp_pr = zeros(p,r); Cbuf = zeros(r,r)

    # Profile objective at GP variance s: RE-FIT β (GLS) and Σ (IW mode) per
    # segment under A = s·R₀(ρ) + diag(1/w), write them into β,Σ,Σi,logdetΣ, and
    # return the total profile marginal log-posterior (+ IG prior on s).
    function fit_at(s)
        tot = iglp(s)
        for k in 1:nseg
            lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
            ts = view(t, lb:ub)
            ldA = 0.0
            for a in 1:p
                ldA = kalman_innov!(view(zcol,1:m), wsk, view(X,lb:ub,a), ts, ρ, s, view(vinv,lb:ub))
                for c in 1:m; ZX[c,a] = zcol[c]; end
            end
            for b in 1:r
                kalman_innov!(view(zcol,1:m), wsk, view(Y,lb:ub,b), ts, ρ, s, view(vinv,lb:ub))
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
            tr = 0.0
            for b in 1:r, c in 1:r; tr += Si[b,c]*RΣ[c,b]; end
            tot += -0.5*(r*ldA + tr + m*logdetΣ[k] + m*r*log(2π))
        end
        tot
    end

    # robust 1-D maximize of the profile over log s: coarse grid bracket + golden
    function search_s()
        bi = 1; bv = -Inf; lstep = (log(s2g_hi)-log(s2g_lo))/(ngrid-1)
        for i in 1:ngrid
            v = fit_at(exp(log(s2g_lo) + (i-1)*lstep))
            (v > bv) && (bv = v; bi = i)
        end
        lo = log(s2g_lo) + max(bi-2,0)*lstep
        hi = log(s2g_lo) + min(bi,ngrid-1)*lstep
        u = _golden_max(uu -> fit_at(exp(uu)), lo, hi)
        s = exp(u); fit_at(s); s        # final fit_at leaves β,Σ at the optimum
    end

    # initial profile at the init breaks
    s2g = gp ? search_s() : (fit_at(0.0); 0.0)
    prev = fill(-1.0, nseg+1); obj = fit_at(s2g)
    @inbounds for it in 1:maxit
        if nb > 0
            # eigendecomposition of Σ (for the break step's rotation only)
            for k in 1:nseg
                Uk = view(Uall,:,:,k); copyto!(Uk, view(Σ,:,:,k))
                dk, _ = LAPACK.syev!('V', 'U', Uk)
                for ℓ in 1:r; dall[ℓ,k] = max(dk[ℓ], 1e-10); end
            end
            # (a) breaks | β, Σ, σ²g — argmax of the split likelihood (G integrated)
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
                    kalman_forward!(wsk, view(rot,1:m,ℓ), ts, ρ, s2g*dv, view(vbuf,1:m))
                    for c in 1:m; Lfwd[c] += wsk.cumll[c]; end
                end
                mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,j+1))
                @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
                mul!(view(rot,1:m,:), view(resid,1:m,:), UR)
                for ℓ in 1:r
                    dv = dR[ℓ]; for c in 1:m; vbuf[c] = dv/w[lb+c-1]; end
                    seg_cumll_bwd!(bwdout, wsk, view(rot,1:m,ℓ), ts, ρ, s2g*dv, view(vbuf,1:m))
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
        end

        # (b) σ²g + β, Σ | breaks — re-profile
        s2g = gp ? search_s() : (fit_at(0.0); 0.0)
        obj = fit_at(s2g)

        # convergence: nb=0 has no break step (one pass suffices); else until the
        # break vector stops moving.
        nb == 0 && break
        same = true; for q in 1:nseg+1; (iv[q] != prev[q]) && (same = false; break); end
        same && break
        for q in 1:nseg+1; prev[q] = iv[q]; end
    end

    # ---- reconstruct G: per band, smoother mean σ²g·R₀ A⁻¹ (Y-Xβ) ----
    G = zeros(n, r)
    if gp
        for k in 1:nseg
            lb = Int(iv[k])+1; ub = Int(iv[k+1]); m = ub-lb+1
            ts = view(t, lb:ub)
            mul!(view(resid,1:m,:), view(X,lb:ub,:), view(β,:,:,k))
            @views resid[1:m,:] .= Y[lb:ub,:] .- resid[1:m,:]
            for b in 1:r
                kalman_smooth_mean!(view(zcol,1:m), wsk, view(resid,1:m,b), ts, ρ, s2g, view(vinv,lb:ub))
                for c in 1:m; G[lb+c-1,b] = zcol[c]; end
            end
        end
    end

    gpfrac = s2g / (1 + s2g)
    S = (β=reshape(copy(β),p,r,nseg,1), iv=reshape(copy(iv),nseg+1,1),
         gpfrac=[gpfrac], σ2g=[s2g],
         Σ=reshape(copy(Σ),r,r,nseg,1), G=reshape(G,n,r,1))
    S, obj
end

# nb selection for the MAP fit by penalized profile marginal log-posterior. The
# (G-integrated) marginal likelihood still rises with nb (each break frees β: p·r
# and Σ: r(r+1)/2 params), so the default penalty is BIC-style: ½·k_seg·log(n) per
# break. Pass a numeric `δ` to force a flat δ·nb penalty. Mirrors
# `fit_adaptive_mv`'s greedy/exhaustive split. Returns (S, nb, loglik).
function fit_map_mv(Y,t,w,X,ρ,pr; maxnb=3, δ=nothing, gp=true, maxit=100, tol=1e-6,
                    greedy=false)
    n = size(Y,1); p = size(X,2); r = size(Y,2)
    pen = δ === nothing ? 0.5*(p*r + (r*(r+1))÷2)*log(n) : float(δ)   # per-break
    kw = (gp=gp, maxit=maxit, tol=tol)
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
