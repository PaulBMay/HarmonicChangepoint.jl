# =============================================================================
# Posterior query of the fitted trend + harmonic features at an arbitrary time.
#
# Both changepoint models write each band's mean as a within-segment trend plus
# a (one- or multi-frequency) harmonic seasonal term. `query_posterior` finds the
# segment a query time falls in and evaluates, per band, the trend LEVEL and the
# Fourier (sin/cos) coefficients of that segment:
#   - pwlr  : value = intercept + slope·t_query   (linear within-segment trend)
#   - UK/OU : value = intercept + g(t_query)       (OU-GP latent trend, gpred)
# Harmonic coefficients are phase/segment quantities and carry no time argument.
#
# Returns a NamedTuple (value, sin, cos):
#   samples=false (default): posterior expectations
#       value :: Vector            length r (bands)
#       sin   :: Matrix            r × nfreqs
#       cos   :: Matrix            r × nfreqs
#   samples=true: the full per-draw posterior (extra trailing sample dimension)
#       value :: Matrix            r × nsamps
#       sin   :: Array{Float64,3}  r × nfreqs × nsamps
#       cos   :: Array{Float64,3}  r × nfreqs × nsamps
# =============================================================================

# OU posterior of the latent trend g at query time tq within a segment:
# continuous-time interpolation of the sampled path (t, tq in days, range ρ).
function gpred(g::AbstractVector, t::AbstractVector, tq::Real, ρ::Real)
    m = length(g)
    tq <= t[1] && return g[1]*exp(-(t[1]-tq)/ρ)
    tq >= t[m] && return g[m]*exp(-(tq-t[m])/ρ)
    i = searchsortedlast(t, tq); φ1 = exp(-(tq-t[i])/ρ); φ2 = exp(-(t[i+1]-tq)/ρ)
    (φ1*(1-φ2^2)*g[i] + φ2*(1-φ1^2)*g[i+1]) / (1-φ1^2*φ2^2)
end

_collapse(value, sinc, cosc) =
    (value = dropdims(mean(value, dims = ndims(value)), dims = ndims(value)),
     sin   = dropdims(mean(sinc,  dims = ndims(sinc)),  dims = ndims(sinc)),
     cos   = dropdims(mean(cosc,  dims = ndims(cosc)),  dims = ndims(cosc)))

# ---- UK / OU-GP fit (output of `ou_gibbs_mv` / `fit_adaptive_mv`) ------------
# Dispatch: the second positional argument is the observation-time vector `t`
# (days, as passed to the fit), `ρ` is the OU range, `tq` the query time.
function query_posterior(S::NamedTuple, t::AbstractVector, ρ::Real, tq::Real;
                         samples::Bool = false)
    p, r, nseg, ns = size(S.β)
    nfreqs = (p - 1) ÷ 2
    val = zeros(r, ns); sinc = zeros(r, nfreqs, ns); cosc = zeros(r, nfreqs, ns)
    for s in 1:ns
        iv = view(S.iv, :, s)
        k = 1
        for j in 1:nseg-1
            (t[Int(iv[j+1])] < tq) && (k += 1)          # advance past breaks before tq
        end
        lb = Int(iv[k]) + 1; ub = Int(iv[k+1])
        for b in 1:r
            val[b, s] = S.β[1, b, k, s] +
                        gpred(view(S.G, lb:ub, b, s), view(t, lb:ub), tq, ρ)
            for f in 1:nfreqs
                sinc[b, f, s] = S.β[2*f,   b, k, s]
                cosc[b, f, s] = S.β[2*f+1, b, k, s]
            end
        end
    end
    samples ? (value = val, sin = sinc, cos = cosc) : _collapse(val, sinc, cosc)
end

# ---- pwlr fit (the `samples` NamedTuple returned by `pwlr`) ------------------
# Dispatch: the second positional argument is the scalar query time `t_query`
# (used for the slope term); `t_idx` is the query's index into the fitted series
# (selects the segment via `intervals`); `nfreqs` matches the harmonic design.
function query_posterior(fit::NamedTuple, t_query::Real, t_idx::Real, nfreqs::Integer;
                         intercept::Bool = true, slope::Bool = true, samples::Bool = false)
    p, r, nseg, ns = size(fit.β)
    coff = 1*intercept + 1*slope + 1                    # first harmonic column
    val = zeros(r, ns); sinc = zeros(r, nfreqs, ns); cosc = zeros(r, nfreqs, ns)
    for i in 1:ns
        ivs = view(fit.intervals, :, i)
        k = clamp(searchsortedlast(ivs, t_idx, 1, nseg+1, Base.Order.Forward), 1, nseg)
        βk = view(fit.β, :, :, k, i)
        for b in 1:r
            v = 0.0
            intercept && (v += βk[1, b])
            slope     && (v += βk[1*intercept + 1, b] * t_query)
            val[b, i] = v
            for f in 1:nfreqs
                ind = coff + 2*(f-1)
                sinc[b, f, i] = βk[ind,   b]
                cosc[b, f, i] = βk[ind+1, b]
            end
        end
    end
    samples ? (value = val, sin = sinc, cos = cosc) : _collapse(val, sinc, cosc)
end
