module LossUtils
# https://nextjournal.com/r3tex/loss-landscape

using Statistics: std
using ProgressLogging: @progress
using ComponentArrays: ComponentArray
using LazyGrids: ndgrid

using ..LuxUtils: getweights

function dict_weights(layers)
    out = Dict{Symbol, Any}()
    for k in keys(layers)
		out[k] = getweights(layers[k])
    end
    return out
end

function dict_weights_randn(layers, n)
    dicts = [Dict{Symbol, Any}() for i in 1:n]
    for dict in dicts, k in keys(layers)
		w, b = layers[k]
		randy = (randn(eltype(w), size(w)), randn(eltype(b), size(b)))
		# dict[k] = filternorm(randy, getweights(layers[k]))
		dict[k] = layernorm(randy, getweights(layers[k]))
    end
    return dicts
end

lnkern(dir, min) = sqrt(sum(min.^2)) .* dir ./ sqrt(sum(dir.^2))

function fnkern(dir, min)
    dim = ndims(dir) == 4 ? [1,2,3] : [1]
    # @show dim ndims(dir)
    sqrt.(sum(min.^2, dims=dim)) .* dir ./ sqrt.(sum(dir.^2, dims=dim))
end

layernorm(dir, min) = lnkern.(dir, min)
filternorm(dir, min) = fnkern.(dir, min)

function linear2x(α, β, θ_center, θ1, θ2)
    ψ = α .* θ1 .+ (1 - α) .* θ_center
    # @show θ_center θ1 θ2 ψ
    # @show layernorm(ψ, θ_opt)
    return β .* θ2 .+ (1 - β) .* layernorm(ψ, θ_center)
end

function barycentric(α, β, θ_center, θ1, θ2)
    ϕ  = α .* (θ1 .- θ_center) .+ θ_center
    ψ  = β .* (θ2 .- θ_center) .+ θ_center
    # @show θ_center θ1 θ2 ψ ϕ
    # @show filternorm(ψ, θ_center)
    return α .* filternorm(ϕ, θ_center) .+ (1 - β) .* filternorm(ψ, θ_center)
end

function simplex(α, β, θ_center, θ1, θ2)
    ϕ  = α .* θ1 .+ (1 - α) .* θ2
    ψ  = β .* ϕ .+ (1 - β) .* θ_center
    return ψ
end

function loss_landscape(controlODE, loss_fun, θ_opt, resolution, interpolation)
    
    # x, y = collect(resolution), collect(resolution)
    X, Y = ndgrid(resolution, resolution)
    # z = zeros(eltype(θ_opt_vec), length(x), length(y))
    z = zeros(eltype(resolution), size(X))
    
    θ_opt_vec = collect(ComponentArray(θ_opt))
    @info "Loss with the parameter set provided" loss_fun(controlODE, θ_opt_vec)
    θ1, θ2 = dict_weights_randn(θ_opt, 2)
    θm = dict_weights(θ_opt)
    θr = Dict()
    # @progress for (i, α) in enumerate(x), (j, β) in enumerate(y)
    @progress for (i, (α, β)) in enumerate(zip(X, Y))
        for k in keys(θm)
            res = interpolation(α, β, θm[k], θ1[k], θ2[k])
            θr[k] = (; weight=res[1], bias=res[2])
        end
        loss = loss_fun(controlODE, ComponentArray(θr))
        # z[i, j] = loss
        z[i] = loss
    end
    # mask inf values
    # z[isinf.(z)] .= maximum(z[.!(isinf.(z))]) + std(z[.!(isinf.(z))])
    # z[isinf.(z)] .= -1.
    # log.(reshape(z, length(x), length(y)))
    # reshape(z, length(x), length(y))
    # log.(z)
    X,Y,z
end

end  # LossUtils
