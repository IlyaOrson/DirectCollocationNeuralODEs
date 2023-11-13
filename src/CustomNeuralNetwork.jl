module CustomNeuralNetwork

using ArgCheck: @argcheck

function dense(x, p, out_size, fun=identity)
	in_size = length(x)
	@argcheck length(p) == (in_size + 1) * out_size
	matrix = reshape(p[1:(out_size * in_size)], out_size, in_size)
	biases = p[(out_size * in_size + 1):end]  # end = out_size * (in_size + 1)
	# @show size(matrix) size(biases)
	return fun.(matrix * x .+ biases)
end

function count_params(state_size, layers_sizes)
	in_size = state_size
	sum = 0
	for out_size in layers_sizes
		# @show in_size out_size
		sum += (in_size + 1) * out_size
		in_size = out_size
	end
	return sum
end

function chain(x, p, sizes, funs)
	@argcheck length(p) == count_params(length(x), sizes)
	state = x
	start_param = 1
	for (out_size, fun) in zip(sizes, funs)
		in_size = length(state)
		nparams_dense_layer = (in_size + 1) * out_size
		# @show start_param insize length(p[start_param : start_param + nparams_dense_layer - 1])
		state = dense(
			state, p[start_param:(start_param + nparams_dense_layer - 1)], out_size, fun
		)
		start_param += nparams_dense_layer
	end
	return state
end

function start_values_sampler(
	state_size, layers_sizes; factor=(in, out) -> 2 / (in + out)
)
	in_size = state_size
	total_params = count_params(state_size, layers_sizes)
	sample_array = zeros(total_params)
	start_param = 1

	for out_size in layers_sizes

		num_weights = in_size * out_size
		num_biases = out_size
		num_params = num_weights + num_biases

		# Xavier initialization: variance = 2/(in+out)
		# factor = 2/(in_size + out_size)
		# V(c*X) = c^2 * V(X)
		# σ(c*X) = c^2 * σ(X)

		samples = factor(in_size, out_size) * randn(num_weights)

		sample_array[start_param:(start_param + num_weights - 1)] = samples

		start_param += num_params
		in_size = out_size
	end
	return sample_array
end

end  # CustomNeuralNetwork
