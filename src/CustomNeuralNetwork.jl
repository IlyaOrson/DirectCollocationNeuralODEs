module CustomNeuralNetwork

using ArgCheck: @argcheck

function custom_dense_layer(input, params, out_size, activation=identity)
	in_size = length(input)
	@argcheck length(params) == (in_size + 1) * out_size
	matrix = reshape(params[1:(out_size * in_size)], out_size, in_size)
	biases = params[(out_size * in_size + 1):end]  # end = out_size * (in_size + 1)
	# @show size(matrix) size(biases)
	return activation.(matrix * input .+ biases)
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

function custom_chain(input, params, sizes, funs)
	@argcheck length(params) == count_params(length(input), sizes)
	state = input
	start_param = 1
	for (out_size, fun) in zip(sizes, funs)
		in_size = length(state)
		nparams_dense_layer = (in_size + 1) * out_size
		# @show start_param insize length(p[start_param : start_param + nparams_dense_layer - 1])
		state = custom_dense_layer(
			state, params[start_param:(start_param + nparams_dense_layer - 1)], out_size, fun
		)
		start_param += nparams_dense_layer
	end
	return state
end

function xavier_sampler(
	state_size, layers_sizes; custom_std=(in, out) -> sqrt(2 / (in + out)) * 5/3
)
	in_size = state_size
	total_params = count_params(state_size, layers_sizes)
	sample_array = zeros(total_params)
	start_param = 1

	for out_size in layers_sizes

		num_weights = in_size * out_size
		num_biases = out_size
		num_params = num_weights + num_biases

		# https://pytorch.org/docs/stable/nn.init.html
		# Xavier initialization: variance = 2/(in+out)
		# tanh gain = 3/2
		# V(c*X) = c^2 * V(X)
		# σ(c*X) = c * σ(X)

		samples = custom_std(in_size, out_size) * randn(num_weights)

		sample_array[start_param:(start_param + num_weights - 1)] = samples

		start_param += num_params
		in_size = out_size
	end
	return sample_array
end

end  # CustomNeuralNetwork
