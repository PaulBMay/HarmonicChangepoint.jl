using HarmonicChangepoint

using LinearAlgebra, Random, Distributions
using Plots

Random.seed!(96)

n = 1_000
r = 2

X = [ones(n) 1:n]

priors = (
    β = (μ=nothing, Q = Diagonal([1.0, 100.0])),
    Σ = (Scale = 10.0*Matrix(I, r, r), df = 20)
)

truparams = simparams(X, priors, 1; σcz = 0.1)
truparams.Σ

w = rand(Poisson(5), n) .+ 1

Y = simdata(X, w, truparams)

scatter(Y[:,1], zcolor = w)

params = deepcopy(truparams)

samples, elpd = pwlr(Y, X, w, priors, params, 10_000; getelpd = true)

mean(samples.β, dims = 4)
truparams.β