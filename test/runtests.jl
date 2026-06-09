using HarmonicChangepoint
using Test
using LinearAlgebra, Random

@testset "HarmonicChangepoint.jl" begin
    # Write your tests here.
end

@testset "UK (OU-GP) changepoint variant" begin
    # linear model must remain alongside the new variant
    @test isdefined(HarmonicChangepoint, :pwlr)
    @test isdefined(HarmonicChangepoint, :ou_gibbs_mv)
    @test isdefined(HarmonicChangepoint, :fit_adaptive_mv)

    # OU Kalman marginal loglik is exact vs the dense GP
    Random.seed!(1)
    t = cumsum(rand(40).*30 .+ 5); n = length(t)
    ρ = 1826.0; σ2g = 0.4; v = 0.05 .+ 0.1.*rand(n); y = randn(n)
    Σ = σ2g.*exp.(.-abs.(t.-t')./ρ) + Diagonal(v)
    dense = -0.5*(dot(y, Σ\y) + logdet(cholesky(Σ)) + n*log(2π))
    @test isapprox(kalman_forward(y, t, ρ, σ2g, v).loglik, dense; atol=1e-8)

    # FFBS posterior mean matches the dense GP posterior mean (MC tolerance)
    K = σ2g.*exp.(.-abs.(t.-t')./ρ)
    post_mean = K*((K+Diagonal(v))\y)
    G = reduce(hcat, ffbs_sample(y, t, ρ, σ2g, v) for _ in 1:5000)
    @test maximum(abs.(vec(sum(G, dims=2)./size(G,2)) .- post_mean)) < 0.05

    # abrupt-step series: a single break is found, near the true change time.
    # times: Apr–Oct (day-of-month 15) for 2000–2025, no Dates dependency.
    Random.seed!(7)
    doy = [105.0,135,166,196,227,258,288]                 # ~15th of Apr..Oct
    ty  = [yr + d/365.25 for yr in 2000:2025 for d in doy]
    dv  = (ty .- 2017).*365.25; nn = length(dv)
    abr(s) = (s < 2010 ? 0.30 : 0.12 + 0.16*(1-exp(-(s-2010)/3.5))) + 0.025*sinpi(2s)
    X = reduce(vcat, ([1.0 sinpi(2s) cospi(2s)] for s in ty)); w = fill(1.0, nn)
    Y = reshape(2 .*(abr.(ty) .+ 0.02 .*randn(nn)) .- 1, :, 1)
    pr = (βQ=Matrix(1.0I,3,3), Σdf=3.0, ΣScale=0.02*Matrix(I,1,1), ag=3.0, bg=0.1)
    S, _ = ou_gibbs_mv(Y, dv, w, X, 5*365.25, 1, pr, 2000, 1000)
    # S.iv[2,:] are integer break POSITIONS; map through dv to get the day/year
    brk_yr = [2017 + dv[Int(i)]/365.25 for i in S.iv[2,:]]
    mean_brk = sum(brk_yr)/length(brk_yr)
    @test 2008.0 < mean_brk < 2011.0
end
