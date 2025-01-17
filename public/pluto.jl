### A Pluto.jl notebook ###
# v0.19.46

#> [frontmatter]
#> title = "Training NeuralODEs with direct collocation"
#> date = "2025-01-17"
#> tags = ["NeuralODEs", "collocation", "training", "julia", "optimal control"]
#> description = "Avoid adjoint calculations!"
#> 
#>     [[frontmatter.author]]
#>     name = "Ilya Orson Sandoval"
#>     url = "https://github.com/ilyaorson"

using Markdown
using InteractiveUtils

# ╔═╡ 07b1f884-6179-483a-8a0b-1771da59799f
@time @time_imports begin
	using PlutoUI
	using Base: @kwdef
	import Statistics
	using LazyGrids: ndgrid
	using Random: Random
	using Lux: Lux, Dense, Chain
	using ComponentArrays: ComponentArray, getaxes
	using Functors: fmap
	using ProgressLogging: @progress
	using LinearAlgebra: norm
	using InfiniteOpt
	using Ipopt: Ipopt
	# using MadNLP: MadNLP
	using ReverseDiff, ForwardDiff, Enzyme
	using QuadGK: quadgk
	using BenchmarkTools: @benchmark

	using ArgCheck: @argcheck, @check
	using Suppressor: @capture_out
	using DataInterpolations: LinearInterpolation

	using SciMLBase: AbstractODEAlgorithm, AbstractODEProblem, ODEProblem, DECallback
	using DiffEqCallbacks: FunctionCallingCallback
	import OrdinaryDiffEq

	using LazyGrids: ndgrid
	using UnicodePlots: Plot, lineplot, lineplot!, histogram, vline!
	using PyPlot: matplotlib, plt, ColorMap
end

# ╔═╡ 3d3ef220-af04-4c72-aabe-58d29ae9eb2b
TableOfContents()

# ╔═╡ ee9852e9-b7a9-4573-a385-45e80f5db1f4
md"# Settings"

# ╔═╡ 4fe5aa44-e7b6-409a-94b5-ae82420e2f69
begin
	# matplotlib.use("agg", force=true)
	matplotlib.style.use(["ggplot","fast"])
	# matplotlib.rcParams["path.simplify"] = false
	matplotlib.get_backend()
end

# ╔═╡ 6aa3bbbe-1b38-425d-9e4e-be173f4ddfdd
begin
	u0 =    [0.0, 1.0]
    tspan = (0.0, 5.0)
end;

# ╔═╡ 321c438f-0e82-4e6a-a6d3-ff119d6bf556
begin
	layer_size = 12
	layer_num = 1

	neural_num_supports = 20
	neural_nodes_per_element = 2
end;

# ╔═╡ edd99d8c-979c-44a4-bac0-1da3412d4bb4
begin
	collocation_num_supports = 20
	collocation_nodes_per_element = 2
end;

# ╔═╡ ccba9574-4a9b-4d41-923d-6897482339db
begin
	opt_tol = 1e-1
end;

# ╔═╡ 2c3e0711-a456-4962-b53b-12d0654704f1
md"## Lux framework"

# ╔═╡ 87503749-df13-40b3-af30-84da22ddf276
function vector_to_parameters(ps_new::AbstractMatrix, ps::NamedTuple)
	@check size(ps_new, 1) == Lux.parameterlength(ps)
	vector_to_parameters(ps_new[:], ps)
end

# ╔═╡ 4d2ea03b-2c57-424e-b8d3-23950904bf8b
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
# vector_to_parameters(weights, ps)
# ComponentArray(weights, getaxes(ComponentArray(ps)))

# ╔═╡ 65378422-0dd7-4e62-b465-7c1dff882784
begin
	# Construct the layer
	final_activation(x) = (tanh.(x) * 1.3 / 2) .+ 0.3
	lchain = Chain(
		Dense(2, layer_size, x -> tanh(x); init_weight=Lux.glorot_normal),
		Dense(layer_size, 1, final_activation; init_weight=Lux.glorot_normal)
	)
end;

# ╔═╡ 1db842a6-5d25-451c-9fab-29b9babbe1bb
begin
	# Seeding
	rng = Random.default_rng()
	# Random.seed!(rng, 0)

	# Parameter and State Variables
	ps, st = Lux.setup(rng, lchain)

	xavier_weights = ComponentArray(ps)

	# Run the model
	# y, st = Lux.apply(lchain, u0, ps, st)
end;

# ╔═╡ cf18c79a-79b1-4e89-900a-c913d55a7f65
xavier_weights |> histogram

# ╔═╡ 13c38797-bc05-4794-b99b-a4aafb2e0503
md"## Custom NN for JuMP usage"

# ╔═╡ c6557e55-f427-4fee-a04c-b18b0e7b8226
function dense(x, p, out_size, fun=identity)
	in_size = length(x)
	@argcheck length(p) == (in_size + 1) * out_size
	matrix = reshape(p[1:(out_size * in_size)], out_size, in_size)
	biases = p[(out_size * in_size + 1):end]  # end = out_size * (in_size + 1)
	# @show size(matrix) size(biases)
	return fun.(matrix * x .+ biases)
end

# ╔═╡ 5e94cf41-6b08-4a8c-9487-9029b0ad70ce
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

# ╔═╡ 33318735-3495-4ae2-a447-3431aca6e557
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

# ╔═╡ 57628f90-d581-4d2b-82dc-f0a9c300d086
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

# ╔═╡ 253458ec-12d1-4228-b57a-73c02b3b2c49
begin
    layer_sizes = ((layer_size for _ in 1:layer_num)..., 1)
    activations = ((tanh for _ in 1:layer_num)..., final_activation)

	# use Lux initialization instead
    # xavier_weights = start_values_sampler(nstates, layer_sizes)

	nstates = length(u0)
	nparams = length(xavier_weights)

	cat_params = vcat(u0 , collect(xavier_weights))
	grad_container =  similar(cat_params)
end;

# ╔═╡ 0ac534b3-40b4-44ef-a903-0ea69db883e2
md"## User defined functions API in JuMP"

# ╔═╡ 02377f26-039d-4685-ba94-1574b3b18aa6
function vector_fun(z)
	# @show z typeof(z) z[1:nstates] z[nstates+1:end]
	x = collect(z[1:nstates])
	p = collect(z[(nstates + 1):end])
	return chain(x, p, layer_sizes, activations)  # custom nn
	# return Lux.apply(lchain, x, vector_to_parameters(p, ps), st)[begin]  # lux nn
end

# ╔═╡ aa575018-f57d-4180-9663-44da68d6c77c
# NOTE: JUMP does not support vector valued functions
# https://jump.dev/JuMP.jl/stable/manual/nlp/#User-defined-functions-with-vector-inputs
function scalar_fun(z...)
	return vector_fun(collect(z))
end

# ╔═╡ 12b71a53-cbe3-4cb3-a383-c341048616e8
md"## Gradient using AD"

# ╔═╡ ceef04c1-7c0b-4e6f-ad2c-3679e6ed0055
begin
	tape =  ReverseDiff.GradientTape(x -> vector_fun(x)[1], cat_params)
	ctape = ReverseDiff.compile(tape)
end

# ╔═╡ cf9abc47-962f-4837-a4e5-7e5984dae474
vector_fun(cat_params)[begin]

# ╔═╡ 09ef9cc4-2564-44fe-be1e-ce75ad189875
grad!(grad_container, params) = ReverseDiff.gradient!(grad_container, ctape, params)

# ╔═╡ 826d4479-9db0-44d6-98ae-b46e7a6a0b4e
md"## Sanity checks"

# ╔═╡ 62626cd0-c55b-4c29-94c2-9d736f335349
md"# Collocation with policy"

# ╔═╡ d8888d92-71df-4c0e-bdc1-1249e3da23d0
function build_neural_collocation(;
	num_supports::Integer,
    nodes_per_element::Integer,
	constrain_states::Bool=true,
)

	optimizer = optimizer_with_attributes(
		Ipopt.Optimizer,
		"print_level" => 4,
		"tol" => opt_tol,
        # "max_iter" => 10_000,
		"hessian_approximation" => "limited-memory",
		# "mu_strategy" => "adaptive",
	)
	# optimizer = optimizer_with_attributes(
	# 	MadNLP.Optimizer,
	# 	"linear_solver" => MadNLP.LapackCPUSolver,
	# 	"tol" => opt_tol,
	# )
	model = InfiniteModel(optimizer)
	method = OrthogonalCollocation(nodes_per_element)

	@infinite_parameter(
		model,
		t in [tspan[1], tspan[2]],
		num_supports = num_supports,
		derivative_method = method
	)
	# @infinite_parameter(model, p[1:nparams] in [-1, 1], independent = true, num_supports = param_supports)

	@variable(model, p[i=1:nparams], start = xavier_weights[i])

	# state variables
	@variables(
		model,
		begin
			x[1:2], Infinite(t)
		end
	)

    # initial conditions
	@constraint(model, [i = 1:2], x[i](0) == u0[i])

    if constrain_states
        @constraint(model, -0.4 <= x[1])  # FIXME
    end

	# https://github.com/jump-dev/MathOptInterface.jl/pull/1819
	# JuMP.register(optimizer_model(model), :scalar_fun, length(cat_params), scalar_fun; autodiff=true)  # forward mode
	JuMP.register(optimizer_model(model), :scalar_fun, length(cat_params), scalar_fun, grad!)  # reverse mode

	@constraints(
		model,
		begin
			∂(x[1], t) == (1 - x[2]^2) * x[1] - x[2] + scalar_fun(vcat(x, p)...)[1]
			∂(x[2], t) == x[1]
		end
	)

	@objective(
		model, Min, integral(x[1]^2 + x[2]^2 + scalar_fun(vcat(x, p)...)[1]^2, t)
	)

	return model
end

# ╔═╡ 2f758dec-f50c-4d75-936d-ded6985862d7
function extract_infopt_results(model; time=:t, state=:x, control=:c, param=:p)
    @argcheck has_values(model)

    model_keys = keys(model.obj_dict)

    times = supports(model[time])
    states = hcat(value.(model[state])...) |> permutedims

    results = (; times=times, states=states)

    if control in model_keys
        controls = hcat(value.(model[control])...) |> permutedims
        results = merge(results, (; controls=controls))
    end
    if param in model_keys
        params = hcat(value.(model[param])...) |> permutedims
        results = merge(results, (; params=params))
    end
    return results
end

# ╔═╡ 6618578d-fbb9-4247-912f-b80baaa99cf7
function optimize_infopt!(infopt_model::InfiniteModel; verbose=false, solver_report=false)

    solver_output = @capture_out InfiniteOpt.optimize!(infopt_model)

    # list possible termination status: model |> termination_status |> typeof
    jump_model = optimizer_model(infopt_model)
    # OPTIMAL = 1, LOCALLY_SOLVED = 4
    if Int(termination_status(jump_model)) ∉ (1, 4)
        @error raw_status(jump_model) termination_status(jump_model)
        error("The collocation optimization failed.")
    else
        @info "Solver summary" solution_summary(jump_model; verbose)
        # @info "Objective value" objective_value(infopt_model)
        solver_report && @info "Solver report" solver_output
    end
    return infopt_model
end

# ╔═╡ 954da4a3-ca0b-42bf-aec6-8ea9aae630b1
# ╠═╡ disabled = true
#=╠═╡

  ╠═╡ =#

# ╔═╡ 732b8e45-fb51-454b-81d2-2d084c12df73
begin
	neural_model = build_neural_collocation(
			num_supports=neural_num_supports,
	    	nodes_per_element=neural_nodes_per_element,
	)
	optimize_infopt!(neural_model; verbose=true)
end

# ╔═╡ edd395e2-58b7-41af-85ae-6af612154df5
result = extract_infopt_results(neural_model);

# ╔═╡ 0ac3a6d6-9354-4944-86a8-205bddc99019
result.params |> histogram

# ╔═╡ bb8ba557-5161-4193-81e3-7ab55b7a5e9c
md"## Neural control interface"

# ╔═╡ 7128af29-3e39-496c-9a1c-3ec287174c9f
struct ControlODE{T<:Real}
    controller
    system
    u0::AbstractVector{T}
    tspan::Tuple{T,T}
    tsteps::Union{Nothing, AbstractVector{T}}
    integrator::AbstractODEAlgorithm
    #sensealg::AbstractSensitivityAlgorithm
    prob::AbstractODEProblem
    inplace::Bool

    function ControlODE(
        controller,
        system,
        u0,
        tspan;
        tsteps::Union{Nothing,AbstractVector{<:Real}}=nothing,
        Δt::Union{Nothing,Real}=nothing,
        npoints::Union{Nothing,Real}=nothing,
        input::Symbol=:state,
        integrator=OrdinaryDiffEq.AutoTsit5(OrdinaryDiffEq.Rosenbrock23()),
        #sensealg=SENSEALG,
    )
        # check tsteps construction
        if !isnothing(tsteps)
            @argcheck tspan[begin] == tsteps[begin]
            @argcheck tspan[end] == tsteps[end]
        elseif !isnothing(Δt)
            tsteps = range(tspan...; step=Δt)
        elseif !isnothing(npoints)
            tsteps = range(tspan...; length=npoints)
        else
			tsteps=nothing
        end

        # check domain types
        time_type = eltype(tspan)
        space_type = eltype(u0)
        @argcheck space_type == time_type

        # construct ODE problem
        @assert length(methods(system)) == 1

        # number of arguments for inplace form system (du, u, p, t, controller; input)
        local prob, inplace
        if methods(system)[1].nargs < 6
            inplace = false
            dudt(u, p, t) = system(u, p, t, controller; input)
            prob = ODEProblem(dudt, u0, tspan)
        else
            inplace = true
            dudt!(du, u, p, t) = system(du, u, p, t, controller; input)
            prob = ODEProblem(dudt!, u0, tspan)
        end
        return new{space_type}(
            controller, system, u0, tspan, tsteps, integrator, prob, inplace
        )
    end
end

# ╔═╡ 25932314-8443-4565-ac36-b9ffde16d620
function solve(code::ControlODE, params; kwargs...)
    return OrdinaryDiffEq.solve(
        code.prob,
        code.integrator;
        p=params,
        #saveat=code.tsteps,  # this should not be necessary
        #sensealg=nothing,  # code.sensealg,
        abstol=1e-1,
        reltol=1e-2,
		verbose=false,
        kwargs...,
    )
end

# ╔═╡ c31f52de-34a0-403f-9683-cc38a2386b62
md"## Van Der Pol system in the interface"

# ╔═╡ 281a5a4d-e156-4e04-a013-d1b1351dd822
begin
	Base.@kwdef struct VanDerPol
		μ=1f0
	end
	function (S::VanDerPol)(du, u, p, t, controller; input=:state)
	    @argcheck input in (:state, :time)

	    # neural network outputs the controls taken by the system
	    x1, x2 = u

	    if input == :state
	        c1 = controller(u, p)[1]  # control based on state and parameters
	    elseif input == :time
	        c1 = controller(t, p)[1]  # control based on time and parameters
	    end

	    # dynamics of the controlled system
	    x1_prime = S.μ * (1 - x2^2) * x1 - x2 + c1
	    x2_prime = x1

	    # update in-place
	    @inbounds begin
	        du[1] = x1_prime
	        du[2] = x2_prime
	    end
	    return nothing
	    # return [x1_prime, x2_prime]
	end
end

# ╔═╡ 58b96a16-f288-4e34-99b7-e0ea11a347e8
state_bounds(S::VanDerPol) = Dict(1 => (-0.4, Inf))

# ╔═╡ 3061a099-3d0d-472d-a1b5-e4785b980014
function state_bound_violation(S::VanDerPol, states::Matrix)
	sb = state_bounds(S)
	for state in eachcol(states)
		for dim in keys(sb)
			lb, up = sb[dim]
			coord = state[dim]
			if coord < lb || coord > up
				# @show coord
				return true
			end
		end
	end
	return false
end

# ╔═╡ 1772d71a-1f7f-43cd-a4ad-0f7f54c960d0
begin
	system = VanDerPol()
	# controller = (x, p) -> chain(x, p, layer_sizes, activations)
	controller = (x, p) -> Lux.apply(lchain, x, vector_to_parameters(p, ps), st)[1]
	controlODE = ControlODE(controller, system, u0, tspan)
end;

# ╔═╡ 2590217d-f9e4-4ddf-a0f8-4830b870fad5
map( # same chain results
	≈,
	Lux.apply(lchain, controlODE.u0, vector_to_parameters(xavier_weights, ps), st)[1],
	chain(controlODE.u0, xavier_weights, layer_sizes, activations)
) |> all

# ╔═╡ d2f60a56-3615-459b-bbbf-9dee822a7213
map( # same derivatives
	≈,
	ReverseDiff.gradient(p -> chain(controlODE.u0, p, layer_sizes, activations), ComponentArray(ps)),
	ReverseDiff.gradient(p -> sum(Lux.apply(lchain, u0, p, st)[1]), ComponentArray(ps))
) |> all

# ╔═╡ e42a34d5-cef0-4630-a19f-cee50b7851a7
@benchmark Lux.apply($lchain, $controlODE.u0, $xavier_weights, $st)

# ╔═╡ 3b23856e-3ebe-4441-a47f-e9078c824d58
@benchmark chain($controlODE.u0, $xavier_weights, $layer_sizes, $activations)

# ╔═╡ c6603841-f09c-49ac-9b84-81295b09b22b
md"## Discrete and continuous losses"

# ╔═╡ 52afbd53-5128-4482-b929-2c71398be122
function loss_discrete(controlODE, params; kwargs...)
    objective = zero(eltype(params))
    sol = solve(controlODE, params; kwargs...)
	sol_arr = Array(sol)
	if state_bound_violation(controlODE.system, sol_arr)
		return Inf
	end
    for (i, col) in enumerate(axes(sol, 2))
		i == 1 && continue
        s = sol_arr[:, col]
        c = controlODE.controller(s, params)
		Δt = sol.t[i] - sol.t[i-1]
        objective += Δt * ( s[1]^2 + s[2]^2 + c[1]^2 )
    end
    return objective
end

# ╔═╡ 6704374c-70de-4e4d-9523-e516c1072348
loss_discrete(controlODE, result.params)

# ╔═╡ d964e018-1e22-44a0-baef-a18ed5979a4c
function loss_continuous(controlODE, params; inf_penalty=true, kwargs...)
	# try
	sol = solve(controlODE, params; kwargs...)
	if inf_penalty && state_bound_violation(controlODE.system, Array(sol))
		return Inf
	end
		result, error = quadgk(
			(t)-> sum(abs2, vcat(sol(t), controlODE.controller(sol(t), params))), controlODE.tspan...,
			rtol=1e-3,
		)
	    return result
	# catch e
	# 	return Inf
	# end
end

# ╔═╡ e47b3510-6e9d-461a-8ca6-ce32d30404ad
# same loss
loss_continuous(controlODE, xavier_weights) ≈
loss_continuous(controlODE, ComponentArray(vector_to_parameters(xavier_weights, ps)))

# ╔═╡ 42f26a4c-ac76-4212-80f9-82858ce2959c
loss_continuous(controlODE, result.params)

# ╔═╡ a7e85595-0b5e-41a7-89f8-939ff7d7d04c
function run_simulation(
    controlODE::ControlODE,
    params;
    control_input=:state,
    noise::Union{Nothing,Real}=nothing,
    vars::Union{Nothing,AbstractArray{<:Integer}}=nothing,
    callback::Union{Nothing,DECallback}=nothing,
    kwargs...,
)
    if !isnothing(noise)
        if !isnothing(callback)
            @warn "Supplied callback will be replaced by a noise callback."
        end

		@argcheck noise >= zero(noise)

		if !isnothing(vars)
        	@argcheck all(var in eachindex(controlODE.u0) for var in vars)
		else
			vars = eachindex(controlODE.u0)
		end

        function noiser(u, t, integrator)
            for var in vars
                u[var] += noise * randn()
            end
        end
        callback = FunctionCallingCallback(
            noiser;
            # funcat=tsteps,
            func_everystep=true,
        )
    end

    # integrate with given parameters
    solution = solve(controlODE, params; callback, kwargs...)

    # construct arrays with the same type used by the integrator
    elements_type = eltype(solution.t)
    states = Array(solution)
    total_steps = size(states, 2)
    # state_dimension = size(states, 1)

    # regenerate controls from controlODE.controller
    if control_input == :state
        control_dimension = length(controlODE.controller(solution.u[begin], params))
        controls = zeros(elements_type, control_dimension, total_steps)
        for (step, state) in enumerate(solution.u)
            controls[:, step] = controlODE.controller(state, params)
        end
    elseif control_input == :time
        control_dimension = length(controlODE.controller(solution.t[begin], params))
        controls = zeros(elements_type, control_dimension, total_steps)
        for (step, time) in enumerate(solution.t)
            controls[:, step] = controlODE.controller(time, params)
        end
    else
        @check control_input in [:state, :time]
    end

    return solution.t, states, controls
end

# ╔═╡ 4733bebb-e27f-47c4-bf6b-23dcf037c70c
begin
	abstract type PhasePlotMarkers end

	@kwdef struct IntegrationPath <: PhasePlotMarkers
	    points
	    fmt = "m:"
	    label = "Integration path"
	    markersize = nothing
	    linewidth = 6
	end

	@kwdef struct InitialMarkers <: PhasePlotMarkers
	    points
	    fmt = "bD"
	    label = "Initial state"
	    markersize = 12
	    linewidth = nothing
	end

	@kwdef struct FinalMarkers <: PhasePlotMarkers
	    points
	    fmt = "r*"
	    label = "Final state"
	    markersize = 18
	    linewidth = nothing
	end

	function states_markers(states_array)
	    start_mark = InitialMarkers(; points=states_array[:, 1])
	    marker_path = IntegrationPath(; points=states_array)
	    final_mark = FinalMarkers(; points=states_array[:, end])
	    # returned order is irrelevant
	    return [marker_path, start_mark, final_mark]
	end

	@kwdef struct ShadeConf
	    indicator::Function
	    cmap = "gray"
	    transparency = 1
	end

	function square_bounds(u0, arista)
	    low_bounds = u0 .- repeat([arista / 2], length(u0))
	    high_bounds = u0 .+ repeat([arista / 2], length(u0))
	    bounds = [(l, h) for (l, h) in zip(low_bounds, high_bounds)]
	    return bounds
	end
end

# ╔═╡ 33ff461e-87cc-40ba-b035-0e0d1dcfe7d5
function phase_portrait(
    controlODE,
    params,
    coord_lims;
    time=0.0f0,
    point_base=controlODE.u0,
    points_per_dim=1000,
    projection=[1, 2],
    markers::Union{Nothing,AbstractVector{<:PhasePlotMarkers}}=nothing,
    start_points=nothing,
    start_points_x=nothing,
    start_points_y=nothing,
    title=nothing,
    shader=nothing,
    kwargs...,
)
    dimension = length(controlODE.u0)

    @argcheck length(projection) == 2
    @argcheck all(ind in 1:dimension for ind in projection)
    @argcheck all(x -> isa(x, Tuple) && length(x) == 2, coord_lims)

    state_dtype = eltype(controlODE.u0)
    function stream_interface(coords...)
        @argcheck length(coords) == dimension
        u = zeros(state_dtype, dimension)
        copyto!(u, coords)
        if controlODE.inplace
            # du = deepcopy(coords)
            du = zeros(state_dtype, dimension)
            controlODE.system(du, u, params, time, controlODE.controller)
            return du
        end
        return controlODE.system(u, params, time, controlODE.controller)
    end

    # evaluate system over each combination of coords in the specified ranges
    # NOTE: float64 is relevant for the conversion to pyplot due to inner
    #       numerical checks of equidistant input in the streamplot function
    ranges = [range(Float64.(lims)...; length=points_per_dim) for lims in coord_lims]
    xpoints, ypoints = collect.(ranges[projection])

    coord_arrays = Vector{Array{state_dtype}}(undef, dimension)
    for ind in 1:dimension
        if ind == projection[1]
            coord_arrays[ind] = xpoints
        elseif ind == projection[2]
            coord_arrays[ind] = ypoints
        else
            coord_arrays[ind] = [point_base[ind]]
        end
    end
    # NOTE: the transpose is required to get f.(a',b) instead of the default f.(a, b')
    # states_grid = stream_interface.(xpoints', ypoints)
    coord_grids = ndgrid(coord_arrays...)
    states_grid = stream_interface.(coord_grids...)

    disposable_dims = Tuple(filter(x -> x ∉ projection, 1:dimension))
    filtered_states_grid = dropdims(states_grid; dims=disposable_dims) |> permutedims

    xphase, yphase = [getindex.(filtered_states_grid, dim) for dim in projection]

    magnitude = map((x, y) -> sqrt(sum(x^2 + y^2)), xphase, yphase)

	# return xpoints, ypoints, xphase, yphase, magnitude

    fig = plt.figure()

    # integration_direction = isnothing(start_points) ? "both" : "forward"
    ax = fig.add_subplot()

    strm = ax.streamplot(
        xpoints,
        ypoints,
        xphase,
        yphase;
        color=magnitude,
        linewidth=1.5,
        density=1.5,
        cmap="summer",
        kwargs...,
    )

    # displaying points (handles multiple points as horizontally concatenated)
    if !isnothing(markers)
        for plotconf in markers
            points_projected = plotconf.points[projection, :]
            ax.plot(
                points_projected[1, :],
                points_projected[2, :],
                plotconf.fmt;
                label=plotconf.label,
                markersize=plotconf.markersize,
                linewidth=plotconf.linewidth,
            )
        end
    end

    xlims, ylims = coord_lims[projection]

    if !isnothing(shader)
        mask =
            dropdims(shader.indicator.(coord_grids...); dims=disposable_dims) |> permutedims
        ax.imshow(
            mask;
            extent=(xlims..., ylims...),
            alpha=shader.transparency,
            cmap=shader.cmap,
            aspect="auto",
        )
    end

    ax.set(; xlim=xlims .+ (-0.05, 0.05), ylim=ylims .+ (-0.05, 0.05))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    !isnothing(title) && ax.set_title(title)

    # fig.colorbar(strm.lines)
    ax.legend(labelspacing=1.2)

    # remove frame
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.spines["left"].set_visible(false)

    plt.tight_layout()

    return fig
end

# ╔═╡ 022af0de-c125-4dd4-82e2-79409e5df76c
function constraint_indicator(coords...)
	if coords[1] > -0.4
		return true
	end
	return false
end

# ╔═╡ 10529d5d-486e-4455-9876-5ac46768ce8a
md"# Collocation without policy"

# ╔═╡ 7eec23a8-1840-47db-aba8-89f9f057d9a4
function build_classic_collocation(
    u0,
    tspan;
    num_supports::Integer,
    nodes_per_element::Integer,
    constrain_states::Bool=true,
)
	optimizer = optimizer_with_attributes(
		Ipopt.Optimizer,
		"print_level" => 4,
		"tol" => opt_tol,
        # "max_iter" => 10_000,
		"hessian_approximation" => "limited-memory",
		# "mu_strategy" => "adaptive",
	)
	# optimizer = optimizer_with_attributes(
	# 	MadNLP.Optimizer,
	# 	"linear_solver" => MadNLP.LapackCPUSolver,
	# 	"tol" => opt_tol,
	# )
    model = InfiniteModel(optimizer)
    method = OrthogonalCollocation(nodes_per_element)
    @infinite_parameter(
        model, t in [tspan[1], tspan[2]], num_supports = num_supports, derivative_method = method
    )

    @variables(
        model,
        begin  # "start" sets the initial guess values
            # state variables
            x[1:2], Infinite(t)
            # control variables
            c[1], Infinite(t)
        end
    )

    # initial conditions
    @constraint(model, [i = 1:2], x[i](0) == u0[i])

    # control range
    @constraint(model, -0.3 <= c[1] <= 1.0)

    if constrain_states
        @constraint(model, -0.4 <= x[1])
    end

    # dynamic equations
    @constraints(
        model,
        begin
            ∂(x[1], t) == (1 - x[2]^2) * x[1] - x[2] + c[1]
            ∂(x[2], t) == x[1]
        end
    )

    @objective(model, Min, integral(x[1]^2 + x[2]^2 + c[1]^2, t))

	return model
end

# ╔═╡ 1a9bcfe7-23f5-4c89-946c-0a97194778f0
begin
	classic_model = build_classic_collocation(
		controlODE.u0,
		controlODE.tspan;
		num_supports=collocation_num_supports,
		nodes_per_element=collocation_nodes_per_element,
	);
	optimize_infopt!(classic_model; verbose=false)
end;

# ╔═╡ 9a9050d3-10ef-4ba9-892f-443dfc782d7c
function interpolant_controller(collocation; plot=nothing)

    num_controls = size(collocation.controls, 1)

    interpolations = [
        LinearInterpolation(collocation.controls[i, :], collocation.times) for i in 1:num_controls
        # CubicSpline(collocation.controls[i, :], collocation.times) for i in 1:num_controls
    ]

    function control_profile(t, p)
        return [interpolations[i](t) for i in 1:num_controls]
	end

    return control_profile
end

# ╔═╡ 57ec210a-0dda-4a1c-9478-736d669b7090
begin
	collocation_results = extract_infopt_results(classic_model)
	times_c, states_c, controls_c = collocation_results
end

# ╔═╡ 751dadcb-9cc7-4719-94f0-33455bdf493d
begin
	reference_controller = interpolant_controller(collocation_results)
	collocationODE = ControlODE(reference_controller, system, u0, tspan; input=:time)
end;

# ╔═╡ b502800d-3011-4a17-b25a-662aa7a1b951
begin
	times_cp, states_cp, controls_cp = run_simulation(collocationODE, nothing; control_input=:time, dt=1f-2)
	lineplot(times_cp, controls_cp[1,:])
end

# ╔═╡ 11f17f4e-10a1-4fa8-bcb9-247e4b39ef47
md"## Querying the optimization model"

# ╔═╡ 8a55733f-8ed8-4eb6-8dbb-cfad02aff2ae
jump_classic_model = classic_model |> optimizer_model

# ╔═╡ 05007017-a435-460b-8051-ee12575785e3
all_parameters(classic_model)[1] |> supports

# ╔═╡ e4c7d0d0-1d6e-4116-a8b7-b59addd2f5e6
# methodswith(typeof(classic_model), InfiniteOpt)

# ╔═╡ bdf615eb-820e-42a3-8e30-12f4a817c994
all_constraints(classic_model)

# ╔═╡ 94f2b440-d233-4396-a978-ad12237201a6
JuMP.objective_function(classic_model)

# ╔═╡ 440a6aa5-89ae-4be9-a50e-e46a31b8b973
all_constraints(jump_classic_model, AffExpr, MOI.EqualTo{Float64})

# ╔═╡ 29eb3b57-fff7-4747-9b1d-b86481499664
all_constraints(neural_model)

# ╔═╡ 55b9cca5-9fc3-44eb-943b-25cd65e59e11
JuMP.objective_function(neural_model)

# ╔═╡ 2fcb5c4e-a5bf-4f26-bede-d0c53db9256d
@time begin
	plt.clf()
    function indicator(coords...)
        if coords[1] > -0.4
            return true
        end
        return false
    end
    shader = ShadeConf(; indicator)
    phase_portrait(
        controlODE,
        result.params[:],
        square_bounds(controlODE.u0, 7);
        shader=ShadeConf(; indicator),
        projection=[1, 2],
        markers=states_markers(states_c),
        title="Optimized policy with constraints",
		# linewidth=1.1,
		# density=0.8,
    )
	plt.gcf()
end

# ╔═╡ 63cb7acb-2e6c-4cea-8938-dc3891b274d3
begin
	rounder(x) = round(x; digits=3)

	obj_neural = objective_value(neural_model) |> rounder
	obj_classic = objective_value(classic_model) |> rounder

	num_vars_classic = value.(all_variables(optimizer_model(classic_model))) |> length
	num_vars_neural = value.(all_variables(optimizer_model(neural_model))) |> length

	time_neural = solve_time(neural_model) |> rounder
	time_classic = solve_time(classic_model) |>rounder

	md"""# Comparison of results
	| Problem | Optima | Time | Vars |
	| --- | --- | --- | --- |
	| Neural | $obj_neural |  $time_neural | $num_vars_neural |
	| Classic   | $obj_classic | $time_classic | $num_vars_classic |"""
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ArgCheck = "dce04be8-c92d-5529-be00-80e4d2c0e197"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
DataInterpolations = "82cc6244-b520-54b8-b5a6-8a565e85f1d0"
DiffEqCallbacks = "459566f4-90b8-5000-8ac3-15dfb0a30def"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
InfiniteOpt = "20393b10-9daf-11e9-18c9-8db751c92c57"
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
LazyGrids = "7031d0ef-c40d-4431-b2f8-61a8d2f650db"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
SciMLBase = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Suppressor = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[compat]
ArgCheck = "~2.3.0"
BenchmarkTools = "~1.5.0"
ComponentArrays = "~0.15.17"
DataInterpolations = "~6.4.1"
DiffEqCallbacks = "~4.0.0"
Enzyme = "~0.13.7"
ForwardDiff = "~0.10.36"
Functors = "~0.4.12"
InfiniteOpt = "~0.5.9"
Ipopt = "~1.6.6"
LazyGrids = "~1.0.0"
Lux = "~1.1.0"
OrdinaryDiffEq = "~6.89.0"
PlutoUI = "~0.7.60"
ProgressLogging = "~0.1.4"
PyPlot = "~2.11.5"
QuadGK = "~2.11.1"
ReverseDiff = "~1.15.3"
SciMLBase = "~2.55.0"
Statistics = "~1.11.1"
Suppressor = "~0.2.8"
UnicodePlots = "~3.6.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "36f67e503a12d0aeaa95f477472b512bf1c81fdb"

[[deps.ADTypes]]
git-tree-sha1 = "eea5d80188827b35333801ef97a40c2ed653b081"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.9.0"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.ASL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6252039f98492252f9e47c312c8ffda0e3b9e78d"
uuid = "ae81ac8f-d209-56e5-92de-9978fef736f9"
version = "0.1.3+0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "b392ede862e506d451fc1616e79aa6f4c673dab8"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.38"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "6a55b747d1812e699320963ffde36f1ebdda4099"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.4"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "3640d077b6dafd64ceb8fd5c1ec76f7ca53bcf76"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.16.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra"]
git-tree-sha1 = "0dd7edaff278e346eb0ca07a7e75c9438408a3ce"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.10.3"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "5a97e67919535d6841172016c9530fd69494e5ec"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "e7c529cc31bb85b97631b922fa2e6baf246f5905"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b5278586822443594ff615963b0c09755771b3e0"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.26.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ComponentArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "Functors", "LinearAlgebra", "PackageExtensionCompat", "StaticArrayInterface", "StaticArraysCore"]
git-tree-sha1 = "bc391f0c19fa242fb6f71794b949e256cfa3772c"
uuid = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
version = "0.15.17"

    [deps.ComponentArrays.extensions]
    ComponentArraysAdaptExt = "Adapt"
    ComponentArraysConstructionBaseExt = "ConstructionBase"
    ComponentArraysGPUArraysExt = "GPUArrays"
    ComponentArraysOptimisersExt = "Optimisers"
    ComponentArraysRecursiveArrayToolsExt = "RecursiveArrayTools"
    ComponentArraysReverseDiffExt = "ReverseDiff"
    ComponentArraysSciMLBaseExt = "SciMLBase"
    ComponentArraysTrackerExt = "Tracker"
    ComponentArraysTruncatedStacktracesExt = "TruncatedStacktraces"
    ComponentArraysZygoteExt = "Zygote"

    [deps.ComponentArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SciMLBase = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    TruncatedStacktraces = "781d530d-4396-4725-bb49-402e4bee1e77"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "b19db3927f0db4151cb86d073689f2428e524576"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataInterpolations]]
deps = ["FindFirstFunctions", "ForwardDiff", "LinearAlgebra", "PrettyTables", "RecipesBase", "Reexport"]
git-tree-sha1 = "1e1a04e3345dee5292b0b89d1f44dd202454e319"
uuid = "82cc6244-b520-54b8-b5a6-8a565e85f1d0"
version = "6.4.1"

    [deps.DataInterpolations.extensions]
    DataInterpolationsChainRulesCoreExt = "ChainRulesCore"
    DataInterpolationsOptimExt = "Optim"
    DataInterpolationsRegularizationToolsExt = "RegularizationTools"
    DataInterpolationsSymbolicsExt = "Symbolics"

    [deps.DataInterpolations.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Optim = "429524aa-4258-5aef-a3af-852621145aeb"
    RegularizationTools = "29dad682-9a27-4bc3-9c72-016788665182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffEqBase]]
deps = ["ArrayInterface", "ConcreteStructs", "DataStructures", "DocStringExtensions", "EnumX", "EnzymeCore", "FastBroadcast", "FastClosures", "ForwardDiff", "FunctionWrappers", "FunctionWrappersWrappers", "LinearAlgebra", "Logging", "Markdown", "MuladdMacro", "Parameters", "PreallocationTools", "PrecompileTools", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SciMLStructures", "Setfield", "Static", "StaticArraysCore", "Statistics", "Tricks", "TruncatedStacktraces"]
git-tree-sha1 = "ada2a9faba0e365dca3cc456b5eca94cf3887ac3"
uuid = "2b5f629d-d688-5b77-993f-72d75c75574e"
version = "6.156.1"

    [deps.DiffEqBase.extensions]
    DiffEqBaseCUDAExt = "CUDA"
    DiffEqBaseChainRulesCoreExt = "ChainRulesCore"
    DiffEqBaseDistributionsExt = "Distributions"
    DiffEqBaseEnzymeExt = ["ChainRulesCore", "Enzyme"]
    DiffEqBaseGeneralizedGeneratedExt = "GeneralizedGenerated"
    DiffEqBaseMPIExt = "MPI"
    DiffEqBaseMeasurementsExt = "Measurements"
    DiffEqBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    DiffEqBaseReverseDiffExt = "ReverseDiff"
    DiffEqBaseSparseArraysExt = "SparseArrays"
    DiffEqBaseTrackerExt = "Tracker"
    DiffEqBaseUnitfulExt = "Unitful"

    [deps.DiffEqBase.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    GeneralizedGenerated = "6b9d7cbe-bcb9-11e9-073f-15a7a543e2eb"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.DiffEqCallbacks]]
deps = ["ConcreteStructs", "DataStructures", "DiffEqBase", "DifferentiationInterface", "Functors", "LinearAlgebra", "Markdown", "RecipesBase", "RecursiveArrayTools", "SciMLBase", "StaticArraysCore"]
git-tree-sha1 = "7f700fa4fb6e55f4672f8218ef228107245a2e9d"
uuid = "459566f4-90b8-5000-8ac3-15dfb0a30def"
version = "4.0.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "Compat", "LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "7e10eebf52857cd441274f74e3a172cad950437f"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.6.5"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = "Enzyme"
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = "ForwardDiff"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = "PolyesterForwardDiff"
    DifferentiationInterfaceReverseDiffExt = "ReverseDiff"
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "19eec8ee5cde1c10aa5e5e0f25d10a3b4c8f8f12"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.15"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "d7477ecdafb813ddee2ae727afa94e9dcb5f3fb0"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.112"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "32a334efa1e11f848a65d1286afdc0b4e4b2941b"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.7"

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

    [deps.Enzyme.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.EnzymeCore]]
git-tree-sha1 = "9c3a42611e525352e9ad5e4134ddca5c692ff209"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.4"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "d62681af4449614972ab959a670b6ec68738e548"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.153+0"

[[deps.ExponentialUtilities]]
deps = ["Adapt", "ArrayInterface", "GPUArraysCore", "GenericSchur", "LinearAlgebra", "PrecompileTools", "Printf", "SparseArrays", "libblastrampoline_jll"]
git-tree-sha1 = "8e18940a5ba7f4ddb41fe2b79b6acaac50880a86"
uuid = "d4d017d3-3776-5f7e-afef-a10c40355c18"
version = "1.26.1"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Expronicon]]
deps = ["MLStyle", "Pkg", "TOML"]
git-tree-sha1 = "fc3951d4d398b5515f91d7fe5d45fc31dccb3c9b"
uuid = "6b7a57c9-7cc1-4fdf-b7f5-e857abae3636"
version = "0.8.5"

[[deps.FastBroadcast]]
deps = ["ArrayInterface", "LinearAlgebra", "Polyester", "Static", "StaticArrayInterface", "StrideArraysCore"]
git-tree-sha1 = "ab1b34570bcdf272899062e1a56285a53ecaae08"
uuid = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
version = "0.3.5"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "fd923962364b645f3719855c88f7074413a6ad92"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.0.2"

[[deps.FastLapackInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "cbf5edddb61a43669710cbc2241bc08b36d9e660"
uuid = "29a986be-02c6-4525-aec4-84b980013641"
version = "2.0.4"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FindFirstFunctions]]
git-tree-sha1 = "670e1d9ceaa4a3161d32fe2d2fb2177f8d78b330"
uuid = "64ca27bc-2ba2-4a57-88aa-44e436879224"
version = "1.4.1"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield", "SparseArrays"]
git-tree-sha1 = "f9219347ebf700e77ca1d48ef84e4a82a6701882"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.24.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "64d8e93700c7a3f28f717d265382d52fac9fa1c1"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.12"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "1d6f290a5eb1201cd63574fbc4440c788d5cb38f"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.27.8"

[[deps.GenericSchur]]
deps = ["LinearAlgebra", "Printf"]
git-tree-sha1 = "af49a0851f8113fcfae2ef5027c6d49d0acec39b"
uuid = "c145ed77-6b09-5dd9-b285-bf645a82121e"
version = "0.5.4"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1dc470db8b1131cfc7fb4c115de89fe391b9e780"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.0"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

[[deps.Hwloc]]
deps = ["CEnum", "Hwloc_jll", "Printf"]
git-tree-sha1 = "6a3d80f31ff87bc94ab22a7b8ec2f263f9a6a583"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "3.3.0"
weakdeps = ["AbstractTrees"]

    [deps.Hwloc.extensions]
    HwlocTrees = "AbstractTrees"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dd3b49277ec2bb2c6b94eb1604d4d0616016f7a6"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.2+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "7c4195be1649ae622304031ed46a2f4df989f1eb"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.24"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InfiniteOpt]]
deps = ["AbstractTrees", "DataStructures", "Distributions", "FastGaussQuadrature", "JuMP", "LeftChildRightSiblingTrees", "LinearAlgebra", "MutableArithmetics", "Reexport", "SpecialFunctions"]
git-tree-sha1 = "4f6d88a3b569782566304517354fdb7b72778ad3"
uuid = "20393b10-9daf-11e9-18c9-8db751c92c57"
version = "0.5.9"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.Ipopt]]
deps = ["Ipopt_jll", "LinearAlgebra", "MathOptInterface", "OpenBLAS32_jll", "PrecompileTools"]
git-tree-sha1 = "92588db78296190d27668a560df3997719fc2a25"
uuid = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
version = "1.6.6"

[[deps.Ipopt_jll]]
deps = ["ASL_jll", "Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MUMPS_seq_jll", "SPRAL_jll", "libblastrampoline_jll"]
git-tree-sha1 = "a0950d209a055b3adb6d29ade5cbdf005a6bd290"
uuid = "9cc047cb-c261-5740-88fc-0cf96f7bdcc7"
version = "300.1400.1600+0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "f389674c99bfcde17dc57454011aa44d5a260a40"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "c95f443d4641b128d626a429b51d5185050135b5"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.23.2"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.KLU]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "07649c499349dad9f08dde4243a4c597064663e9"
uuid = "ef3ab10e-7fda-4108-b977-705223b18434"
version = "0.6.0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "5126765c5847f74758c411c994312052eb7117ef"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.27"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "267dad6b4b7b5d529c76d40ff48d33f7e94cb834"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.9.6"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Requires", "Unicode"]
git-tree-sha1 = "4ad43cb0a4bb5e5b1506e1d1f48646d7e0c80363"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.2"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "360f6039babd6e4d6364eff0d4fc9120834a2d9a"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.2.1"

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

    [deps.LazyArrays.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyGrids]]
deps = ["Statistics"]
git-tree-sha1 = "1f1bf025f03f644e6c18f458db813fe2e3a68b5d"
uuid = "7031d0ef-c40d-4431-b2f8-61a8d2f650db"
version = "1.0.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LinearSolve]]
deps = ["ArrayInterface", "ChainRulesCore", "ConcreteStructs", "DocStringExtensions", "EnumX", "FastLapackInterface", "GPUArraysCore", "InteractiveUtils", "KLU", "Krylov", "LazyArrays", "Libdl", "LinearAlgebra", "MKL_jll", "Markdown", "PrecompileTools", "Preferences", "RecursiveFactorization", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "SparseArrays", "Sparspak", "StaticArraysCore", "UnPack"]
git-tree-sha1 = "8941ad4bdd83768359801982e143008349b1a827"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "2.35.0"

    [deps.LinearSolve.extensions]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCUDAExt = "CUDA"
    LinearSolveCUDSSExt = "CUDSS"
    LinearSolveEnzymeExt = "EnzymeCore"
    LinearSolveFastAlmostBandedMatricesExt = "FastAlmostBandedMatrices"
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMetalExt = "Metal"
    LinearSolvePardisoExt = "Pardiso"
    LinearSolveRecursiveArrayToolsExt = "RecursiveArrayTools"

    [deps.LinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "8084c25a250e00ae427a379a5b607e7aed96a2dd"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.171"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.LossFunctions]]
deps = ["Markdown", "Requires", "Statistics"]
git-tree-sha1 = "8073538845227d3acf8a0c149bf0e150e60d0ced"
uuid = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
version = "0.11.2"

    [deps.LossFunctions.extensions]
    LossFunctionsCategoricalArraysExt = "CategoricalArrays"

    [deps.LossFunctions.weakdeps]
    CategoricalArrays = "324d7699-5711-5eae-9e2f-1d82baa6b597"

[[deps.Lux]]
deps = ["ADTypes", "Adapt", "ArgCheck", "ArrayInterface", "ChainRulesCore", "Compat", "ConcreteStructs", "DispatchDoctor", "EnzymeCore", "FastClosures", "ForwardDiff", "Functors", "GPUArraysCore", "LinearAlgebra", "LossFunctions", "LuxCore", "LuxLib", "MLDataDevices", "MacroTools", "Markdown", "NNlib", "Optimisers", "Preferences", "Random", "Reexport", "SIMDTypes", "Setfield", "Static", "StaticArraysCore", "Statistics", "VectorizationBase", "WeightInitializers"]
git-tree-sha1 = "bd1ad2abb5ba0c5b5ff3ce076c1434ccb4873939"
uuid = "b2108857-7c20-44ae-9111-449ecde12c47"
version = "1.1.0"

    [deps.Lux.extensions]
    LuxComponentArraysExt = "ComponentArrays"
    LuxEnzymeExt = "Enzyme"
    LuxFluxExt = "Flux"
    LuxMLUtilsExt = "MLUtils"
    LuxMPIExt = "MPI"
    LuxMPINCCLExt = ["CUDA", "MPI", "NCCL"]
    LuxReverseDiffExt = ["FunctionWrappers", "ReverseDiff"]
    LuxSimpleChainsExt = "SimpleChains"
    LuxTrackerExt = "Tracker"
    LuxZygoteExt = "Zygote"

    [deps.Lux.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    FunctionWrappers = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SimpleChains = "de6bee2f-e2f4-4ec7-b6ed-219cc6f6e9e5"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxCore]]
deps = ["Compat", "DispatchDoctor", "Random"]
git-tree-sha1 = "1c1e81a6d0613d13ea9f7dff7291ea407624d74a"
uuid = "bb33d45b-7691-41d6-9220-0943567d0623"
version = "1.0.1"

    [deps.LuxCore.extensions]
    LuxCoreArrayInterfaceReverseDiffExt = ["ArrayInterface", "ReverseDiff"]
    LuxCoreArrayInterfaceTrackerExt = ["ArrayInterface", "Tracker"]
    LuxCoreChainRulesCoreExt = "ChainRulesCore"
    LuxCoreEnzymeCoreExt = "EnzymeCore"
    LuxCoreFunctorsExt = "Functors"
    LuxCoreMLDataDevicesExt = "MLDataDevices"
    LuxCoreSetfieldExt = "Setfield"

    [deps.LuxCore.weakdeps]
    ArrayInterface = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Setfield = "efcf1570-3423-57d1-acb7-fd33fddbac46"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.LuxLib]]
deps = ["ArrayInterface", "ChainRulesCore", "Compat", "CpuId", "DispatchDoctor", "EnzymeCore", "FastClosures", "ForwardDiff", "Hwloc", "KernelAbstractions", "LinearAlgebra", "LoopVectorization", "LuxCore", "MLDataDevices", "Markdown", "NNlib", "Octavian", "Polyester", "Random", "Reexport", "SLEEFPirates", "Static", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "3abebfd9e4f3217e4e1c2608ed707f6abbd46e9d"
uuid = "82251201-b29d-42c6-8e01-566dec8acb11"
version = "1.3.2"

    [deps.LuxLib.extensions]
    LuxLibAppleAccelerateExt = "AppleAccelerate"
    LuxLibBLISBLASExt = "BLISBLAS"
    LuxLibCUDAExt = "CUDA"
    LuxLibEnzymeExt = "Enzyme"
    LuxLibMKLExt = "MKL"
    LuxLibReverseDiffExt = "ReverseDiff"
    LuxLibTrackerAMDGPUExt = ["AMDGPU", "Tracker"]
    LuxLibTrackerExt = "Tracker"
    LuxLibcuDNNExt = ["CUDA", "cuDNN"]

    [deps.LuxLib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AppleAccelerate = "13e28ba4-7ad8-5781-acae-3021b1ed3924"
    BLISBLAS = "6f275bd8-fec0-4d39-945b-7e95a765fa1e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    MKL = "33e6dc65-8f57-5167-99aa-e5a354878fb2"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1fd0a97409e418b78c53fac671cf4622efdf0f21"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.2+0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random"]
git-tree-sha1 = "e16288e37e76d68c3f1c418e0a2bec88d98d55fc"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.2.0"

    [deps.MLDataDevices.extensions]
    MLDataDevicesAMDGPUExt = "AMDGPU"
    MLDataDevicesCUDAExt = "CUDA"
    MLDataDevicesChainRulesCoreExt = "ChainRulesCore"
    MLDataDevicesFillArraysExt = "FillArrays"
    MLDataDevicesGPUArraysExt = "GPUArrays"
    MLDataDevicesMLUtilsExt = "MLUtils"
    MLDataDevicesMetalExt = ["GPUArrays", "Metal"]
    MLDataDevicesReactantExt = "Reactant"
    MLDataDevicesRecursiveArrayToolsExt = "RecursiveArrayTools"
    MLDataDevicesReverseDiffExt = "ReverseDiff"
    MLDataDevicesSparseArraysExt = "SparseArrays"
    MLDataDevicesTrackerExt = "Tracker"
    MLDataDevicesZygoteExt = "Zygote"
    MLDataDevicescuDNNExt = ["CUDA", "cuDNN"]
    MLDataDevicesoneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MUMPS_seq_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "libblastrampoline_jll"]
git-tree-sha1 = "85047ac569761e3387717480a38a61d2a67df45c"
uuid = "d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"
version = "500.700.300+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MarchingCubes]]
deps = ["PrecompileTools", "StaticArrays"]
git-tree-sha1 = "27d162f37cc29de047b527dab11a826dd3a650ad"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "5b246fca5420ae176d65ed43a2d0ee5897775216"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.31.2"

[[deps.MaybeInplace]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "54e2fdc38130c05b42be423e90da3bade29b74bd"
uuid = "bb5d69b7-63fc-4a16-80bd-7e42200c7bdb"
version = "0.1.4"
weakdeps = ["SparseArrays"]

    [deps.MaybeInplace.extensions]
    MaybeInplaceSparseArraysExt = "SparseArrays"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3eba928678787843e504c153a9b8e80d7d73ab17"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.5.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "da09a1e112fd75f9af2a5229323f01b56ec96a4c"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.24"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.NonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "DiffEqBase", "FastBroadcast", "FastClosures", "FiniteDiff", "ForwardDiff", "LazyArrays", "LineSearches", "LinearAlgebra", "LinearSolve", "MaybeInplace", "PrecompileTools", "Preferences", "Printf", "RecursiveArrayTools", "Reexport", "SciMLBase", "SimpleNonlinearSolve", "SparseArrays", "SparseDiffTools", "StaticArraysCore", "SymbolicIndexingInterface", "TimerOutputs"]
git-tree-sha1 = "bcd8812e751326ff1d4b2dd50764b93df51f143b"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "3.14.0"

    [deps.NonlinearSolve.extensions]
    NonlinearSolveBandedMatricesExt = "BandedMatrices"
    NonlinearSolveFastLevenbergMarquardtExt = "FastLevenbergMarquardt"
    NonlinearSolveFixedPointAccelerationExt = "FixedPointAcceleration"
    NonlinearSolveLeastSquaresOptimExt = "LeastSquaresOptim"
    NonlinearSolveMINPACKExt = "MINPACK"
    NonlinearSolveNLSolversExt = "NLSolvers"
    NonlinearSolveNLsolveExt = "NLsolve"
    NonlinearSolveSIAMFANLEquationsExt = "SIAMFANLEquations"
    NonlinearSolveSpeedMappingExt = "SpeedMapping"
    NonlinearSolveSymbolicsExt = "Symbolics"
    NonlinearSolveZygoteExt = "Zygote"

    [deps.NonlinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    FastLevenbergMarquardt = "7a0df574-e128-4d35-8cbd-3d84502bf7ce"
    FixedPointAcceleration = "817d07cb-a79a-5c30-9a31-890123675176"
    LeastSquaresOptim = "0fc2ff8b-aaa3-5acd-a817-1944a5e08891"
    MINPACK = "4854310b-de5a-5eb6-a2a5-c1dee2bd17f9"
    NLSolvers = "337daf1e-9722-11e9-073e-8b9effe078ba"
    NLsolve = "2774e3e8-f4cf-5e23-947b-6d7e65073b56"
    SIAMFANLEquations = "084e46ad-d928-497d-ad5e-07fa361a48c4"
    SpeedMapping = "f1835b91-879b-4a3f-a438-e4baacf14412"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "7249afa1c4dfd86bfbcc9b28939ab6ef844f4e11"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.4.2"

[[deps.Octavian]]
deps = ["CPUSummary", "IfElse", "LoopVectorization", "ManualMemory", "PolyesterWeave", "PrecompileTools", "Static", "StaticArrayInterface", "ThreadingUtilities", "VectorizationBase"]
git-tree-sha1 = "92410e147bdcaf9e2f982a7cc9b1341fc5dd1a77"
uuid = "6fd5a793-0b7e-452c-907f-f8bfe9c57db4"
version = "0.3.28"

    [deps.Octavian.extensions]
    ForwardDiffExt = "ForwardDiff"
    HyperDualNumbersExt = "HyperDualNumbers"

    [deps.Octavian.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    HyperDualNumbers = "50ceba7f-c3ee-5a84-a6e8-3ad40456ec97"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dd806c813429ff09878ea3eeb317818f3ca02871"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.28+3"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6572fe0c5b74431aaeb0b18a4aa5ef03c84678be"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.3.3"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.OrdinaryDiffEq]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "EnumX", "ExponentialUtilities", "FastBroadcast", "FastClosures", "FillArrays", "FiniteDiff", "ForwardDiff", "FunctionWrappersWrappers", "InteractiveUtils", "LineSearches", "LinearAlgebra", "LinearSolve", "Logging", "MacroTools", "MuladdMacro", "NonlinearSolve", "OrdinaryDiffEqAdamsBashforthMoulton", "OrdinaryDiffEqBDF", "OrdinaryDiffEqCore", "OrdinaryDiffEqDefault", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqExplicitRK", "OrdinaryDiffEqExponentialRK", "OrdinaryDiffEqExtrapolation", "OrdinaryDiffEqFIRK", "OrdinaryDiffEqFeagin", "OrdinaryDiffEqFunctionMap", "OrdinaryDiffEqHighOrderRK", "OrdinaryDiffEqIMEXMultistep", "OrdinaryDiffEqLinear", "OrdinaryDiffEqLowOrderRK", "OrdinaryDiffEqLowStorageRK", "OrdinaryDiffEqNonlinearSolve", "OrdinaryDiffEqNordsieck", "OrdinaryDiffEqPDIRK", "OrdinaryDiffEqPRK", "OrdinaryDiffEqQPRK", "OrdinaryDiffEqRKN", "OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqSDIRK", "OrdinaryDiffEqSSPRK", "OrdinaryDiffEqStabilizedIRK", "OrdinaryDiffEqStabilizedRK", "OrdinaryDiffEqSymplecticRK", "OrdinaryDiffEqTsit5", "OrdinaryDiffEqVerner", "Polyester", "PreallocationTools", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SciMLStructures", "SimpleNonlinearSolve", "SimpleUnPack", "SparseArrays", "SparseDiffTools", "Static", "StaticArrayInterface", "StaticArrays", "TruncatedStacktraces"]
git-tree-sha1 = "cd892f12371c287dc50d6ad3af075b088b6f2d48"
uuid = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
version = "6.89.0"

[[deps.OrdinaryDiffEqAdamsBashforthMoulton]]
deps = ["ADTypes", "DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqLowOrderRK", "Polyester", "RecursiveArrayTools", "Reexport", "Static"]
git-tree-sha1 = "8e3c5978d0531a961f70d2f2730d1d16ed3bbd12"
uuid = "89bda076-bce5-4f1c-845f-551c83cdda9a"
version = "1.1.0"

[[deps.OrdinaryDiffEqBDF]]
deps = ["ArrayInterface", "DiffEqBase", "FastBroadcast", "LinearAlgebra", "MacroTools", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "OrdinaryDiffEqSDIRK", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "StaticArrays", "TruncatedStacktraces"]
git-tree-sha1 = "b4498d40bf35da0b6d22652ff2e9d8820590b3c6"
uuid = "6ad6398a-0878-4a85-9266-38940aa047c8"
version = "1.1.2"

[[deps.OrdinaryDiffEqCore]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "DataStructures", "DiffEqBase", "DocStringExtensions", "EnumX", "FastBroadcast", "FastClosures", "FillArrays", "FunctionWrappersWrappers", "InteractiveUtils", "LinearAlgebra", "Logging", "MacroTools", "MuladdMacro", "Polyester", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators", "SciMLStructures", "SimpleUnPack", "Static", "StaticArrayInterface", "StaticArraysCore", "TruncatedStacktraces"]
git-tree-sha1 = "5595eb94d0dd3cde2f5cf456394a1e0a61237e95"
uuid = "bbf590c4-e513-4bbe-9b18-05decba2e5d8"
version = "1.6.0"
weakdeps = ["EnzymeCore"]

    [deps.OrdinaryDiffEqCore.extensions]
    OrdinaryDiffEqCoreEnzymeCoreExt = "EnzymeCore"

[[deps.OrdinaryDiffEqDefault]]
deps = ["DiffEqBase", "EnumX", "LinearAlgebra", "LinearSolve", "OrdinaryDiffEqBDF", "OrdinaryDiffEqCore", "OrdinaryDiffEqRosenbrock", "OrdinaryDiffEqTsit5", "OrdinaryDiffEqVerner", "PrecompileTools", "Preferences", "Reexport"]
git-tree-sha1 = "c8223e487d58bef28a3535b33ddf8ffdb44f46fb"
uuid = "50262376-6c5a-4cf5-baba-aaf4f84d72d7"
version = "1.1.0"

[[deps.OrdinaryDiffEqDifferentiation]]
deps = ["ADTypes", "ArrayInterface", "DiffEqBase", "FastBroadcast", "FiniteDiff", "ForwardDiff", "FunctionWrappersWrappers", "LinearAlgebra", "LinearSolve", "OrdinaryDiffEqCore", "SciMLBase", "SparseArrays", "SparseDiffTools", "StaticArrayInterface", "StaticArrays"]
git-tree-sha1 = "e63ec633b1efa99e3caa2e26a01faaa88ba6cef9"
uuid = "4302a76b-040a-498a-8c04-15b101fed76b"
version = "1.1.0"

[[deps.OrdinaryDiffEqExplicitRK]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "RecursiveArrayTools", "Reexport", "TruncatedStacktraces"]
git-tree-sha1 = "4dbce3f9e6974567082ce5176e21aab0224a69e9"
uuid = "9286f039-9fbf-40e8-bf65-aa933bdc4db0"
version = "1.1.0"

[[deps.OrdinaryDiffEqExponentialRK]]
deps = ["DiffEqBase", "ExponentialUtilities", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqSDIRK", "OrdinaryDiffEqVerner", "RecursiveArrayTools", "Reexport", "SciMLBase"]
git-tree-sha1 = "f63938b8e9e5d3a05815defb3ebdbdcf61ec0a74"
uuid = "e0540318-69ee-4070-8777-9e2de6de23de"
version = "1.1.0"

[[deps.OrdinaryDiffEqExtrapolation]]
deps = ["DiffEqBase", "FastBroadcast", "LinearSolve", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "Polyester", "RecursiveArrayTools", "Reexport"]
git-tree-sha1 = "fea595528a160ed5cade9eee217a9691b1d97714"
uuid = "becaefa8-8ca2-5cf9-886d-c06f3d2bd2c4"
version = "1.1.0"

[[deps.OrdinaryDiffEqFIRK]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "LinearSolve", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "RecursiveArrayTools", "Reexport", "SciMLOperators"]
git-tree-sha1 = "795221c662698851328cb7787965ab4a180d9468"
uuid = "5960d6e9-dd7a-4743-88e7-cf307b64f125"
version = "1.1.1"

[[deps.OrdinaryDiffEqFeagin]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "Polyester", "RecursiveArrayTools", "Reexport", "Static"]
git-tree-sha1 = "a7cc74d3433db98e59dc3d58bc28174c6c290adf"
uuid = "101fe9f7-ebb6-4678-b671-3a81e7194747"
version = "1.1.0"

[[deps.OrdinaryDiffEqFunctionMap]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "RecursiveArrayTools", "Reexport", "SciMLBase", "Static"]
git-tree-sha1 = "925a91583d1ab84f1f0fea121be1abf1179c5926"
uuid = "d3585ca7-f5d3-4ba6-8057-292ed1abd90f"
version = "1.1.1"

[[deps.OrdinaryDiffEqHighOrderRK]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "RecursiveArrayTools", "Reexport", "Static"]
git-tree-sha1 = "103e017ff186ac39d731904045781c9bacfca2b0"
uuid = "d28bc4f8-55e1-4f49-af69-84c1a99f0f58"
version = "1.1.0"

[[deps.OrdinaryDiffEqIMEXMultistep]]
deps = ["DiffEqBase", "FastBroadcast", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "Reexport"]
git-tree-sha1 = "9f8f52aad2399d7714b400ff9d203254b0a89c4a"
uuid = "9f002381-b378-40b7-97a6-27a27c83f129"
version = "1.1.0"

[[deps.OrdinaryDiffEqLinear]]
deps = ["DiffEqBase", "ExponentialUtilities", "LinearAlgebra", "OrdinaryDiffEqCore", "OrdinaryDiffEqTsit5", "OrdinaryDiffEqVerner", "RecursiveArrayTools", "Reexport", "SciMLBase", "SciMLOperators"]
git-tree-sha1 = "0f81a77ede3da0dc714ea61e81c76b25db4ab87a"
uuid = "521117fe-8c41-49f8-b3b6-30780b3f0fb5"
version = "1.1.0"

[[deps.OrdinaryDiffEqLowOrderRK]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "RecursiveArrayTools", "Reexport", "SciMLBase", "Static"]
git-tree-sha1 = "d4bb32e09d6b68ce2eb45fb81001eab46f60717a"
uuid = "1344f307-1e59-4825-a18e-ace9aa3fa4c6"
version = "1.2.0"

[[deps.OrdinaryDiffEqLowStorageRK]]
deps = ["Adapt", "DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "Polyester", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "Static", "StaticArrays"]
git-tree-sha1 = "590561f3af623d5485d070b4d7044f8854535f5a"
uuid = "b0944070-b475-4768-8dec-fb6eb410534d"
version = "1.2.1"

[[deps.OrdinaryDiffEqNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "DiffEqBase", "FastBroadcast", "FastClosures", "ForwardDiff", "LinearAlgebra", "LinearSolve", "MuladdMacro", "NonlinearSolve", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "PreallocationTools", "RecursiveArrayTools", "SciMLBase", "SciMLOperators", "SciMLStructures", "SimpleNonlinearSolve", "StaticArrays"]
git-tree-sha1 = "a2a4119f3e35f7982f78e17beea7b12485d179e9"
uuid = "127b3ac7-2247-4354-8eb6-78cf4e7c58e8"
version = "1.2.1"

[[deps.OrdinaryDiffEqNordsieck]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqTsit5", "Polyester", "RecursiveArrayTools", "Reexport", "Static"]
git-tree-sha1 = "ef44754f10e0dfb9bb55ded382afed44cd94ab57"
uuid = "c9986a66-5c92-4813-8696-a7ec84c806c8"
version = "1.1.0"

[[deps.OrdinaryDiffEqPDIRK]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "Polyester", "Reexport", "StaticArrays"]
git-tree-sha1 = "a8b7f8107c477e07c6a6c00d1d66cac68b801bbc"
uuid = "5dd0a6cf-3d4b-4314-aa06-06d4e299bc89"
version = "1.1.0"

[[deps.OrdinaryDiffEqPRK]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "Polyester", "Reexport"]
git-tree-sha1 = "da525d277962a1b76102c79f30cb0c31e13fe5b9"
uuid = "5b33eab2-c0f1-4480-b2c3-94bc1e80bda1"
version = "1.1.0"

[[deps.OrdinaryDiffEqQPRK]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "RecursiveArrayTools", "Reexport", "Static"]
git-tree-sha1 = "332f9d17d0229218f66a73492162267359ba85e9"
uuid = "04162be5-8125-4266-98ed-640baecc6514"
version = "1.1.0"

[[deps.OrdinaryDiffEqRKN]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "Polyester", "RecursiveArrayTools", "Reexport"]
git-tree-sha1 = "41c09d9c20877546490f907d8dffdd52690dd65f"
uuid = "af6ede74-add8-4cfd-b1df-9a4dbb109d7a"
version = "1.1.0"

[[deps.OrdinaryDiffEqRosenbrock]]
deps = ["ADTypes", "DiffEqBase", "FastBroadcast", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "LinearSolve", "MacroTools", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "Polyester", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "Static"]
git-tree-sha1 = "96b47cdd12cb4ce8f70d701b49f855271a462bd4"
uuid = "43230ef6-c299-4910-a778-202eb28ce4ce"
version = "1.2.0"

[[deps.OrdinaryDiffEqSDIRK]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "MacroTools", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "RecursiveArrayTools", "Reexport", "SciMLBase", "TruncatedStacktraces"]
git-tree-sha1 = "f6683803a58de600ab7a26d2f49411c9923e9721"
uuid = "2d112036-d095-4a1e-ab9a-08536f3ecdbf"
version = "1.1.0"

[[deps.OrdinaryDiffEqSSPRK]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "Polyester", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "Static", "StaticArrays"]
git-tree-sha1 = "7dbe4ac56f930df5e9abd003cedb54e25cbbea86"
uuid = "669c94d9-1f4b-4b64-b377-1aa079aa2388"
version = "1.2.0"

[[deps.OrdinaryDiffEqStabilizedIRK]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "OrdinaryDiffEqDifferentiation", "OrdinaryDiffEqNonlinearSolve", "RecursiveArrayTools", "Reexport", "StaticArrays"]
git-tree-sha1 = "348fd6def9a88518715425025eadd58517017325"
uuid = "e3e12d00-db14-5390-b879-ac3dd2ef6296"
version = "1.1.0"

[[deps.OrdinaryDiffEqStabilizedRK]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "RecursiveArrayTools", "Reexport", "StaticArrays"]
git-tree-sha1 = "1b0d894c880e25f7d0b022d7257638cf8ce5b311"
uuid = "358294b1-0aab-51c3-aafe-ad5ab194a2ad"
version = "1.1.0"

[[deps.OrdinaryDiffEqSymplecticRK]]
deps = ["DiffEqBase", "FastBroadcast", "MuladdMacro", "OrdinaryDiffEqCore", "Polyester", "RecursiveArrayTools", "Reexport"]
git-tree-sha1 = "4e8b8c8b81df3df17e2eb4603115db3b30a88235"
uuid = "fa646aed-7ef9-47eb-84c4-9443fc8cbfa8"
version = "1.1.0"

[[deps.OrdinaryDiffEqTsit5]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "Static", "TruncatedStacktraces"]
git-tree-sha1 = "96552f7d4619fabab4038a29ed37dd55e9eb513a"
uuid = "b1df2697-797e-41e3-8120-5422d3b24e4a"
version = "1.1.0"

[[deps.OrdinaryDiffEqVerner]]
deps = ["DiffEqBase", "FastBroadcast", "LinearAlgebra", "MuladdMacro", "OrdinaryDiffEqCore", "Polyester", "PrecompileTools", "Preferences", "RecursiveArrayTools", "Reexport", "Static", "TruncatedStacktraces"]
git-tree-sha1 = "81d7841e73e385b9925d5c8e4427f2adcdda55db"
uuid = "79d7bb75-1356-48c1-b8c0-6832512096c2"
version = "1.1.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "6d38fea02d983051776a856b7df75b30cf9a3c1f"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.16"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff"]
git-tree-sha1 = "6c62ce45f268f3f958821a1e5192cf91c75ae89c"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.24"
weakdeps = ["ReverseDiff"]

    [deps.PreallocationTools.extensions]
    PreallocationToolsReverseDiffExt = "ReverseDiff"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[deps.PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "0371ca706e3f295481cbf94c8c36692b072285c2"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.5"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"
weakdeps = ["Enzyme"]

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "b034171b93aebc81b3e1890a036d13a9c4a9e3e0"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.27.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "PrecompileTools", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "6db1a75507051bc18bfa131fbc7c3f169cc4b2f6"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.23"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ReverseDiff]]
deps = ["ChainRulesCore", "DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "cc6cd622481ea366bb9067859446a8b01d92b468"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.15.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "04c968137612c4a5629fa531334bb81ad5680f00"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.13"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.SPRAL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "Libdl", "METIS_jll", "libblastrampoline_jll"]
git-tree-sha1 = "11f3da4b25efacd1cec8e263421f2a9003a5e8e0"
uuid = "319450e9-13b8-58e8-aa9f-8fd1420848ab"
version = "2024.5.8+0"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "Expronicon", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "885904799812ed88a607c03b38e9c3a4a5b08853"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.55.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseMakieExt = "Makie"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra", "MacroTools"]
git-tree-sha1 = "e39c5f217f9aca640c8e27ab21acf557a3967db5"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.10"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLStructures]]
deps = ["ArrayInterface"]
git-tree-sha1 = "25514a6f200219cd1073e4ff23a6324e4a7efe64"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SimpleNonlinearSolve]]
deps = ["ADTypes", "ArrayInterface", "ConcreteStructs", "DiffEqBase", "DiffResults", "DifferentiationInterface", "FastClosures", "FiniteDiff", "ForwardDiff", "LinearAlgebra", "MaybeInplace", "PrecompileTools", "Reexport", "SciMLBase", "Setfield", "StaticArraysCore"]
git-tree-sha1 = "44021f3efc023be3871195d8ad98b865001a2fa1"
uuid = "727e6d20-b764-4bd8-a329-72de5adea6c7"
version = "1.12.3"

    [deps.SimpleNonlinearSolve.extensions]
    SimpleNonlinearSolveChainRulesCoreExt = "ChainRulesCore"
    SimpleNonlinearSolveReverseDiffExt = "ReverseDiff"
    SimpleNonlinearSolveTrackerExt = "Tracker"
    SimpleNonlinearSolveZygoteExt = "Zygote"

    [deps.SimpleNonlinearSolve.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleUnPack]]
git-tree-sha1 = "58e6353e72cde29b90a69527e56df1b5c3d8c437"
uuid = "ce78b400-467f-4804-87d8-8f486da07d0a"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseDiffTools]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "Compat", "DataStructures", "FiniteDiff", "ForwardDiff", "Graphs", "LinearAlgebra", "PackageExtensionCompat", "Random", "Reexport", "SciMLOperators", "Setfield", "SparseArrays", "StaticArrayInterface", "StaticArrays", "Tricks", "UnPack", "VertexSafeGraphs"]
git-tree-sha1 = "7db12cef226aaa8ca730040c9c35e32b86a69b83"
uuid = "47a9eef4-7e08-11e9-0b38-333d64bd3804"
version = "2.22.0"

    [deps.SparseDiffTools.extensions]
    SparseDiffToolsEnzymeExt = "Enzyme"
    SparseDiffToolsPolyesterExt = "Polyester"
    SparseDiffToolsPolyesterForwardDiffExt = "PolyesterForwardDiff"
    SparseDiffToolsSymbolicsExt = "Symbolics"
    SparseDiffToolsZygoteExt = "Zygote"

    [deps.SparseDiffTools.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Sparspak]]
deps = ["Libdl", "LinearAlgebra", "Logging", "OffsetArrays", "Printf", "SparseArrays", "Test"]
git-tree-sha1 = "342cf4b449c299d8d1ceaf00b7a49f4fbc7940e7"
uuid = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
version = "0.3.9"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eeafab08ae20c62c44c8399ccb9354a04b80db50"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.7"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "f35f6ab602df8413a50c4a25ca14de821e8605fb"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.Suppressor]]
deps = ["Logging"]
git-tree-sha1 = "6dbb5b635c5437c68c28c2ac9e39b87138f37c0a"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.8"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "0225f7c62f5f78db35aae6abb2e5cabe38ce578f"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.31"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "5a13ae8a41237cff5ecf34f73eb1b8f42fff6531"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.24"

[[deps.TranscodingStreams]]
git-tree-sha1 = "e84b3a11b9bece70d14cce63406bbc79ed3464d2"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.2"

[[deps.TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "be986ad9dac14888ba338c2554dcfec6939e1393"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.2.1"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodePlots]]
deps = ["ColorSchemes", "ColorTypes", "Contour", "Crayons", "Dates", "LinearAlgebra", "MarchingCubes", "NaNMath", "PrecompileTools", "Printf", "Requires", "SparseArrays", "StaticArrays", "StatsBase"]
git-tree-sha1 = "30646456e889c18fb3c23e58b2fc5da23644f752"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "3.6.4"

    [deps.UnicodePlots.extensions]
    FreeTypeExt = ["FileIO", "FreeType"]
    ImageInTerminalExt = "ImageInTerminal"
    IntervalSetsExt = "IntervalSets"
    TermExt = "Term"
    UnitfulExt = "Unitful"

    [deps.UnicodePlots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    FreeType = "b38be410-82b0-50bf-ab77-7b57e271db43"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Term = "22787eb5-b846-44ae-b979-8e399b8463ab"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "2d17fabcd17e67d7625ce9c531fb9f40b7c42ce4"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.2.1"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "e7f5b81c65eb858bed630fe006837b935518aca5"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.70"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.VertexSafeGraphs]]
deps = ["Graphs"]
git-tree-sha1 = "8351f8d73d7e880bfc042a8b6922684ebeafb35c"
uuid = "19fa3120-7c27-5ec5-8db8-b0b0aa330d6f"
version = "0.2.0"

[[deps.WeightInitializers]]
deps = ["ArgCheck", "ConcreteStructs", "GPUArraysCore", "LinearAlgebra", "Random", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "ed1cfb44422eaa9804c8a596a8227a6e2baaeef7"
uuid = "d49dbf32-c5c2-4618-8acc-27bb2598ef2d"
version = "1.0.3"

    [deps.WeightInitializers.extensions]
    WeightInitializersAMDGPUExt = ["AMDGPU", "GPUArrays"]
    WeightInitializersCUDAExt = ["CUDA", "GPUArrays"]
    WeightInitializersChainRulesCoreExt = "ChainRulesCore"
    WeightInitializersGPUArraysExt = "GPUArrays"
    WeightInitializersMetalExt = ["Metal", "GPUArrays"]
    WeightInitializersoneAPIExt = ["oneAPI", "GPUArrays"]

    [deps.WeightInitializers.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═07b1f884-6179-483a-8a0b-1771da59799f
# ╠═3d3ef220-af04-4c72-aabe-58d29ae9eb2b
# ╟─ee9852e9-b7a9-4573-a385-45e80f5db1f4
# ╠═4fe5aa44-e7b6-409a-94b5-ae82420e2f69
# ╠═6aa3bbbe-1b38-425d-9e4e-be173f4ddfdd
# ╠═321c438f-0e82-4e6a-a6d3-ff119d6bf556
# ╠═edd99d8c-979c-44a4-bac0-1da3412d4bb4
# ╠═ccba9574-4a9b-4d41-923d-6897482339db
# ╟─2c3e0711-a456-4962-b53b-12d0654704f1
# ╟─87503749-df13-40b3-af30-84da22ddf276
# ╟─4d2ea03b-2c57-424e-b8d3-23950904bf8b
# ╠═65378422-0dd7-4e62-b465-7c1dff882784
# ╠═1db842a6-5d25-451c-9fab-29b9babbe1bb
# ╠═cf18c79a-79b1-4e89-900a-c913d55a7f65
# ╟─13c38797-bc05-4794-b99b-a4aafb2e0503
# ╟─c6557e55-f427-4fee-a04c-b18b0e7b8226
# ╟─5e94cf41-6b08-4a8c-9487-9029b0ad70ce
# ╟─33318735-3495-4ae2-a447-3431aca6e557
# ╠═57628f90-d581-4d2b-82dc-f0a9c300d086
# ╠═253458ec-12d1-4228-b57a-73c02b3b2c49
# ╟─0ac534b3-40b4-44ef-a903-0ea69db883e2
# ╠═02377f26-039d-4685-ba94-1574b3b18aa6
# ╠═aa575018-f57d-4180-9663-44da68d6c77c
# ╟─12b71a53-cbe3-4cb3-a383-c341048616e8
# ╠═ceef04c1-7c0b-4e6f-ad2c-3679e6ed0055
# ╠═cf9abc47-962f-4837-a4e5-7e5984dae474
# ╠═09ef9cc4-2564-44fe-be1e-ce75ad189875
# ╟─826d4479-9db0-44d6-98ae-b46e7a6a0b4e
# ╠═e47b3510-6e9d-461a-8ca6-ce32d30404ad
# ╠═2590217d-f9e4-4ddf-a0f8-4830b870fad5
# ╠═d2f60a56-3615-459b-bbbf-9dee822a7213
# ╠═e42a34d5-cef0-4630-a19f-cee50b7851a7
# ╠═3b23856e-3ebe-4441-a47f-e9078c824d58
# ╟─62626cd0-c55b-4c29-94c2-9d736f335349
# ╠═d8888d92-71df-4c0e-bdc1-1249e3da23d0
# ╟─2f758dec-f50c-4d75-936d-ded6985862d7
# ╟─6618578d-fbb9-4247-912f-b80baaa99cf7
# ╠═954da4a3-ca0b-42bf-aec6-8ea9aae630b1
# ╠═732b8e45-fb51-454b-81d2-2d084c12df73
# ╠═edd395e2-58b7-41af-85ae-6af612154df5
# ╠═0ac3a6d6-9354-4944-86a8-205bddc99019
# ╟─bb8ba557-5161-4193-81e3-7ab55b7a5e9c
# ╠═7128af29-3e39-496c-9a1c-3ec287174c9f
# ╠═25932314-8443-4565-ac36-b9ffde16d620
# ╟─c31f52de-34a0-403f-9683-cc38a2386b62
# ╠═281a5a4d-e156-4e04-a013-d1b1351dd822
# ╠═58b96a16-f288-4e34-99b7-e0ea11a347e8
# ╟─3061a099-3d0d-472d-a1b5-e4785b980014
# ╠═1772d71a-1f7f-43cd-a4ad-0f7f54c960d0
# ╟─c6603841-f09c-49ac-9b84-81295b09b22b
# ╠═52afbd53-5128-4482-b929-2c71398be122
# ╠═6704374c-70de-4e4d-9523-e516c1072348
# ╠═d964e018-1e22-44a0-baef-a18ed5979a4c
# ╠═42f26a4c-ac76-4212-80f9-82858ce2959c
# ╠═a7e85595-0b5e-41a7-89f8-939ff7d7d04c
# ╟─4733bebb-e27f-47c4-bf6b-23dcf037c70c
# ╠═33ff461e-87cc-40ba-b035-0e0d1dcfe7d5
# ╟─022af0de-c125-4dd4-82e2-79409e5df76c
# ╟─10529d5d-486e-4455-9876-5ac46768ce8a
# ╠═7eec23a8-1840-47db-aba8-89f9f057d9a4
# ╠═1a9bcfe7-23f5-4c89-946c-0a97194778f0
# ╠═9a9050d3-10ef-4ba9-892f-443dfc782d7c
# ╠═57ec210a-0dda-4a1c-9478-736d669b7090
# ╠═751dadcb-9cc7-4719-94f0-33455bdf493d
# ╠═b502800d-3011-4a17-b25a-662aa7a1b951
# ╟─11f17f4e-10a1-4fa8-bcb9-247e4b39ef47
# ╠═8a55733f-8ed8-4eb6-8dbb-cfad02aff2ae
# ╠═05007017-a435-460b-8051-ee12575785e3
# ╠═e4c7d0d0-1d6e-4116-a8b7-b59addd2f5e6
# ╠═bdf615eb-820e-42a3-8e30-12f4a817c994
# ╠═94f2b440-d233-4396-a978-ad12237201a6
# ╠═440a6aa5-89ae-4be9-a50e-e46a31b8b973
# ╠═29eb3b57-fff7-4747-9b1d-b86481499664
# ╠═55b9cca5-9fc3-44eb-943b-25cd65e59e11
# ╠═2fcb5c4e-a5bf-4f26-bede-d0c53db9256d
# ╟─63cb7acb-2e6c-4cea-8938-dc3891b274d3
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
