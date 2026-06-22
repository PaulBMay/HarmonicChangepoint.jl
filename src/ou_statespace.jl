# =============================================================================
# Universal-kriging changepoint variant — OU state-space primitives.
#
# This is a SEPARATE trend model from `pwlr` (the piecewise-linear-harmonic
# model in functions.jl). Here the within-segment trend is a non-parametric
# Ornstein-Uhlenbeck (continuous-time AR(1) / Matérn-½) Gaussian process rather
# than a fixed-rank linear/harmonic design, so an abrupt within-segment jump is
# forced onto a changepoint instead of being absorbed by a flexible basis.
#
# Single-band / rotated-band scalar building blocks:
#   - kalman_forward : filtering arrays + per-step & cumulative marginal loglik
#   - ffbs_sample    : forward-filter backward-sample of the latent OU trend g
#   - seg_cumll_fwd / seg_cumll_bwd : segment marginal logliks for all split
#     points in O(n) (forward from lb, time-reversed backward from ub)
#
# Model on (residual) data:  y_i = g_i + e_i,  e_i ~ N(0, v_i),
#   g OU with marginal var σ²g, range ρ (days):  g_1~N(0,σ²g),
#   g_i|g_{i-1} ~ N(φ_i g_{i-1}, σ²g(1-φ_i²)),  φ_i = exp(-Δt_i/ρ).
# OU is time-reversible & stationary, so the reversed sequence uses identical
# transitions -> the backward pass gives suffix marginal likelihoods. The Markov
# structure gives a tridiagonal precision -> everything below is O(n).
# =============================================================================

# Forward Kalman filter. Returns NamedTuple with filtering arrays and logliks.
# t: sorted times (days). y: residual obs. v: per-obs noise variance.
function kalman_forward(y::AbstractVector, t::AbstractVector, ρ::Real, σ2g::Real, v::AbstractVector)
    n = length(y)
    a = zeros(n); R = zeros(n); m = zeros(n); P = zeros(n)
    cumll = zeros(n); ll = 0.0
    @inbounds for i in 1:n
        if i == 1
            a[i] = 0.0; R[i] = σ2g
        else
            φ = exp(-(t[i]-t[i-1])/ρ)
            a[i] = φ*m[i-1]
            R[i] = φ^2*P[i-1] + σ2g*(1-φ^2)
        end
        f = y[i] - a[i]; S = R[i] + v[i]
        ll += -0.5*(log(2π*S) + f^2/S)
        cumll[i] = ll
        K = R[i]/S
        m[i] = a[i] + K*f
        P[i] = (1-K)*R[i]
    end
    (a=a, R=R, m=m, P=P, cumll=cumll, loglik=ll)
end

# FFBS: one posterior draw of the latent trend g (length n).
function ffbs_sample(y, t, ρ, σ2g, v; rng=Random.default_rng())
    n = length(y); kf = kalman_forward(y, t, ρ, σ2g, v)
    g = zeros(n)
    g[n] = kf.m[n] + sqrt(kf.P[n])*randn(rng)
    @inbounds for i in n-1:-1:1
        φ = exp(-(t[i+1]-t[i])/ρ)
        J = φ*kf.P[i]/kf.R[i+1]
        mb = kf.m[i] + J*(g[i+1] - kf.a[i+1])
        Pb = kf.P[i] - J^2*kf.R[i+1]
        g[i] = mb + sqrt(max(Pb,0.0))*randn(rng)
    end
    g
end

# Cumulative segment marginal loglik for a segment STARTING at index 1:
# cumll[c] = log p(y_{1..c}). (forward from lb after slicing)
seg_cumll_fwd(y, t, ρ, σ2g, v) = kalman_forward(y, t, ρ, σ2g, v).cumll

# Suffix segment marginal loglik for a segment ENDING at index n:
# out[s] = log p(y_{s..n})  (segment restarts fresh at s). Time-reversed filter.
function seg_cumll_bwd(y, t, ρ, σ2g, v)
    n = length(y)
    cr = kalman_forward(reverse(y), reverse(t).*-1 .+ (t[1]+t[end]), ρ, σ2g, reverse(v)).cumll
    # cr[k] = log p(reversed_{1..k}) = log p(y_{n-k+1 .. n}); map to out[s=n-k+1]
    out = zeros(n)
    @inbounds for k in 1:n; out[n-k+1] = cr[k]; end
    out
end

# -------------------- allocation-free hot-loop variants ----------------------
# Reusable length-n scratch for the scalar OU recursions. The Gibbs sweep calls
# the Kalman filter/FFBS O(bands × segments) times per iteration; routing all of
# them through one OUWork removes the per-call array allocations.
struct OUWork
    a::Vector{Float64}; R::Vector{Float64}; m::Vector{Float64}
    P::Vector{Float64}; cumll::Vector{Float64}
    yr::Vector{Float64}; tr::Vector{Float64}; vr::Vector{Float64}   # reversed scratch (bwd)
end
OUWork(n::Int) = OUWork((zeros(n) for _ in 1:8)...)

# Forward filter into ws (uses entries 1:length(y)); returns total loglik. The
# per-step cumulative loglik lands in ws.cumll. y, t, v may be views.
function kalman_forward!(ws::OUWork, y, t, ρ, σ2g, v)
    n = length(y); a=ws.a; R=ws.R; m=ws.m; P=ws.P; cumll=ws.cumll
    ll = 0.0
    @inbounds for i in 1:n
        if i == 1
            a[i] = 0.0; R[i] = σ2g
        else
            φ = exp(-(t[i]-t[i-1])/ρ)
            a[i] = φ*m[i-1]; R[i] = φ^2*P[i-1] + σ2g*(1-φ^2)
        end
        f = y[i] - a[i]; S = R[i] + v[i]
        ll += -0.5*(log(2π*S) + f^2/S); cumll[i] = ll
        K = R[i]/S; m[i] = a[i] + K*f; P[i] = (1-K)*R[i]
    end
    ll
end

# FFBS draw into g[1:length(y)] using ws as scratch.
function ffbs_sample!(g, ws::OUWork, y, t, ρ, σ2g, v; rng=Random.default_rng())
    n = length(y); kalman_forward!(ws, y, t, ρ, σ2g, v)
    a=ws.a; R=ws.R; m=ws.m; P=ws.P
    @inbounds begin
        g[n] = m[n] + sqrt(P[n])*randn(rng)
        for i in n-1:-1:1
            φ = exp(-(t[i+1]-t[i])/ρ); J = φ*P[i]/R[i+1]
            mb = m[i] + J*(g[i+1] - a[i+1]); Pb = P[i] - J^2*R[i+1]
            g[i] = mb + sqrt(max(Pb,0.0))*randn(rng)
        end
    end
    g
end

# RTS smoother POSTERIOR MEAN of the latent trend g into g[1:length(y)] (the
# deterministic analog of ffbs_sample!: same forward filter + backward recursion
# but without the innovation draw). Used by the MAP/ICM fit.
function kalman_smooth_mean!(g, ws::OUWork, y, t, ρ, σ2g, v)
    n = length(y); kalman_forward!(ws, y, t, ρ, σ2g, v)
    a=ws.a; R=ws.R; m=ws.m; P=ws.P
    @inbounds begin
        g[n] = m[n]
        for i in n-1:-1:1
            φ = exp(-(t[i+1]-t[i])/ρ); J = φ*P[i]/R[i+1]
            g[i] = m[i] + J*(g[i+1] - a[i+1])
        end
    end
    g
end

# Suffix marginal loglik into out[1:length(y)] (segment ENDING at n), via the
# time-reversed filter; reversed series staged in ws.yr/tr/vr.
function seg_cumll_bwd!(out, ws::OUWork, y, t, ρ, σ2g, v)
    n = length(y); t1 = t[1]; tn = t[n]
    @inbounds for k in 1:n
        out[k] = y[n-k+1]                         # stage reversed y in out briefly
    end
    @inbounds for k in 1:n
        ws.yr[k] = out[k]; ws.tr[k] = -t[n-k+1] + (t1+tn); ws.vr[k] = v[n-k+1]
    end
    kalman_forward!(ws, view(ws.yr,1:n), view(ws.tr,1:n), ρ, σ2g, view(ws.vr,1:n))
    @inbounds for k in 1:n; out[n-k+1] = ws.cumll[k]; end
    out
end
