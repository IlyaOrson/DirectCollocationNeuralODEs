module LossUtils
# https://nextjournal.com/r3tex/loss-landscape

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

end  # LossUtils
