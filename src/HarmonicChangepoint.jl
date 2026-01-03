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

end
