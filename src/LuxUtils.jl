module LuxUtils

using ArgCheck: @check
using Lux: Lux
using Functors: fmap

getweights(l) = (l.weight, l.bias)

function vector_to_parameters(ps_new::AbstractMatrix, ps::NamedTuple)
	@check size(ps_new, 1) == Lux.parameterlength(ps)
	vector_to_parameters(ps_new[:], ps)
end

function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @check length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

end  # LuxUtils
