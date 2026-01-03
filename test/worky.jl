# using HarmonicChangepoint

# using LinearAlgebra, Random, Distributions
# using Plots

# Random.seed!(96)

# n = 60
# r = 5
# nbreaks = 1
# nsegments = nbreaks + 1

# time = 1:30:(n*30)

# X = harmonicdesign(time, 365.25, 1)
# p = size(X,2)

# priors = (
#     β = (μ=nothing, Q = Diagonal([1.0, 1_000.0, 0.25, 0.25])),
#     Σ = (Scale = 50.0*Matrix(I, r, r), df = 20)
# )

# truparams = simparams(X, priors, 1; σcz = 0.1)
# truparams.Σ

# w = rand(Poisson(5), n) .+ 1
# #w .= 1

# Y = simdata(X, w, truparams)

# scatter(Y[:,1], zcolor = w)

# params = deepcopy(truparams)

# pwlr(Y, X, w, priors, params, 100; getelpd = true)
# @btime pwlr($Y, $X, $w, $priors, $params, 1_000; getelpd = true)

# mean(samples.intervals, dims = 2)
# truparams.intervals

# # mean(samples.β, dims = 4)
# # truparams.β


# ###############

# buffer = (
#     XtX = zeros(p,p),
#     XtY = zeros(p,r),
#     XtYU = zeros(p,r),
#     Qp = zeros(p,p),
#     U = zeros(r,r,nsegments),
#     d = zeros(r,nsegments),
#     βtilde = zeros(p,r),
#     μp = zeros(p),
#     z = zeros(p),
#     Scalep = zeros(r,r),
#     resid = zeros(n,r),
#     residU = zeros(n,r),
#     lll = zeros(n),
#     llr = zeros(n),
#     logits = zeros(n),
#     probs = zeros(n)
# )

# Ys = sqrt.(w) .* Y
# Xs = sqrt.(w) .* X

# params = deepcopy(truparams)

# HarmonicChangepoint.sampleβ!(params, buffer, Ys, Xs, priors)
# @btime samplec2!(params, buffer, $Ys, $Xs)
# @btime samplec3!(params, $Ys, $Xs)

# params.intervals
# truparams.intervals

# ld = zeros(n)

# @time HarmonicChangepoint.logdensity!(ld, params, buffer, Ys, Xs)

# ##################


# function samplec2!(params, buffer, Y, X)
#     # Dimensions
#     nbreaks = length(params.intervals) - 2
#     r = size(Y,2)
#     # Allocation buffers
#     U = buffer.U
#     d = buffer.d
#     resid = buffer.resid
#     residU = buffer.residU
#     lll = buffer.lll
#     llr = buffer.llr
#     logits = buffer.logits
#     probs = buffer.probs
#     # Loop through breaks
#     @views for j in 1:nbreaks
#         # Subset relevant quants
#         lb = params.intervals[j] + 1
#         ub = params.intervals[j+2]
#         indj = lb:ub

#         Yj = Y[indj,:]
#         Xj = X[indj,:]
#         residj = resid[indj,:]
#         residUj = residU[indj,:]
#         lllj = lll[indj]
#         llrj = llr[indj]
#         logitsj = logits[lb:(ub-1)]
#         probsj = probs[lb:(ub-1)]
#         βleft = params.β[:,:,j]
#         βright = params.β[:,:,j+1]
#         Uleft = U[:,:,j]
#         Uright = U[:,:,j+1]
#         dleft = d[:,j]
#         dright = d[:,j+1]
#         # Left log-likes
#         mul!(residj, Xj, βleft, -1.0, 0.0) 
#         residj .+= Yj
#         mul!(residUj, residj, Uleft)
#         residUj .^= 2
#         for ℓ in 1:r
#             @. residUj[:,ℓ] /= dleft[ℓ]
#         end
#         lllj .=  vec(sum(residUj, dims = 2)) .+ sum(log.(dleft))
#         lllj .*= -0.5
#         # Right log-likes
#         mul!(residj, Xj, βright, -1.0, 0.0) 
#         residj .+= Yj
#         mul!(residUj, residj, Uright)
#         residUj .^= 2
#         for ℓ in 1:r
#             @. residUj[:,ℓ] /= dright[ℓ]
#         end
#         llrj .=  vec(sum(residUj, dims = 2)) .+ sum(log.(dright))
#         llrj .*= -0.5
#         HarmonicChangepoint.bicumsum!(logitsj, lllj, llrj)
#         softmax!(probsj, logitsj)
#         params.intervals[j+1] = indj[HarmonicChangepoint.rcat(probsj)]

#     end

# end

# function samplec3!(params, Y, X)
#     # Dimensions
#     nbreaks = length(params.intervals) - 2
#     r = size(Y,2)

#     for j in 1:nbreaks

#         lb = params.intervals[j] + 1
#         ub = params.intervals[j+2]
#         indj = lb:ub
#         Yj = Y[indj,:]
#         Xj = X[indj,:]

#         βleft = params.β[:,:,j]
#         βright = params.β[:,:,j+1]
#         Σleft = params.Σ[:,:,j]
#         Σright = params.Σ[:,:,j+1]

#         resid = Yj - Xj*βleft
#         Σchol = cholesky(Σleft)
#         lll = -0.5*(
#             vec( sum((resid .* (resid / Σchol)).^2, dims = 2) ) .+
#             logdet(Σchol)
#         )
#         resid = Yj - Xj*βright
#         Σchol = cholesky(Σright)
#         llr = -0.5*(
#             vec( sum((resid .* (resid / Σchol)).^2, dims = 2) ) .+
#             logdet(Σchol)
#         )

#         logits = zeros(length(indj)-1)
#         HarmonicChangepoint.bicumsum!(logits, lll, llr)
#         probs = softmax(logits)
#         params.intervals[j+1] = indj[rand(Categorical(probs))]


#     end

# end