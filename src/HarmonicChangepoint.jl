module HarmonicChangepoint

using LinearAlgebra
using LinearAlgebra.LAPACK
using Random, Distributions
using StatsFuns: logsumexp, softmax!

include("functions.jl")
export pwlr
export simparams
export simdata
export harmonicdesign

# Universal-kriging (OU-GP) changepoint variant — a SEPARATE trend model from
# pwlr; the within-segment trend is a non-parametric OU Gaussian process.
include("ou_statespace.jl")
include("ou_changepoint.jl")
export ou_gibbs_mv
export fit_adaptive_mv
export kalman_forward
export ffbs_sample

# Posterior query of trend value + harmonic coefficients at an arbitrary time
# (works for both the pwlr and UK/OU-GP fits).
include("query.jl")
export query_posterior
export gpred

end
