module HarmonicChangepoint

using LinearAlgebra
using Random, Distributions
using StatsFuns: logsumexp

include("functions.jl")
export pwlr
export simparams
export simdata
export harmonicdesign

end
