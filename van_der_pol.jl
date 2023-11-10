### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 31b5c73e-9641-11ec-2b0b-cbd62716cc97
@time begin
	import Pkg

	# activate the shared project environment
	Pkg.activate(Base.current_project())

	# instantiate, i.e. make sure that all packages are downloaded
	Pkg.instantiate()
end

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
	# return chain(x, p, layer_sizes, activations)  # custom nn
	return Lux.apply(lchain, x, vector_to_parameters(p, ps), st)[begin]  # lux nn
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

# ╔═╡ 7559fb6d-b98b-45bb-af12-d088fd74f18c
md"""# Loss landscape

https://nextjournal.com/r3tex/loss-landscape
"""

# ╔═╡ dec21a44-f25d-4a56-bb3e-8b24f913626b
md"## Normalizations"

# ╔═╡ 08ecf45f-bd51-46c4-94f0-a438a0397c64
getweights(l) = (l.weight, l.bias)

# ╔═╡ ae6dbe7f-32dd-438c-be54-ee4d2abee7fd
function dict_weights(layers)
    out = Dict{Symbol, Any}()
    for k in keys(layers)
		out[k] = getweights(layers[k])
    end
    return out
end

# ╔═╡ 54b8fb32-0a6d-4a1b-a40e-d0ffd1eedbf0
begin  # https://nextjournal.com/r3tex/loss-landscape
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
end

# ╔═╡ af9cd30d-a8b8-4ef1-83aa-df7aa9036940
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

# ╔═╡ 9e3c1ec0-6fa8-4842-98f6-44f26e0dc76b
dict_weights_randn(vector_to_parameters(result.params[:], ps), 2);

# ╔═╡ df32b1d0-3d65-4797-9715-500a6e70e2b6
dict_weights(vector_to_parameters(result.params[:], ps));

# ╔═╡ 2526091c-a56f-4e0b-90f4-79ffa634c436
function landscape(controlODE, θ_opt_vec, resolution, interpolate; pnt = ps)
    x, y = collect(resolution), collect(resolution)
	z = zeros(eltype(θ_opt_vec), length(x), length(y))
    θ_opt = vector_to_parameters(θ_opt_vec, pnt)
	θ1, θ2 = dict_weights_randn(θ_opt, 2)
    θm = dict_weights(θ_opt)
	θr = Dict()
    @progress for (i, α) in enumerate(x), (j, β) in enumerate(y)
        for k in keys(θm)
            res = interpolate(α, β, θm[k], θ1[k], θ2[k])
			θr[k] = (; weight=res[1], bias=res[2])
        end
		# lss = loss_continuous(controlODE, ComponentArray(θr))
		lss = loss_discrete(controlODE, ComponentArray(θr))
		z[i, j] = lss
	end
	@show Statistics.std(z[.!(isinf.(z))])
    z[isinf.(z)] .= maximum(z[.!(isinf.(z))]) + Statistics.std(z[.!(isinf.(z))])
    # log.(reshape(z, length(x), length(y)))
    # reshape(z, length(x), length(y))
	# log.(z)
	z
end

# ╔═╡ f8716793-b97d-4058-b5a6-8e68d00313b9
vector_to_parameters(result.params[:], ps)

# ╔═╡ b1ae1940-6b4a-4a10-bb46-2e692c85c2d3
@time begin
	resolution = -1f0:1f-1:1f1
	# zmap = landscape(controlODE, xavier_weights[:], resolution, linear2x)
	# zmap = landscape(controlODE, xavier_weights[:], resolution, barycentric)
	# zmap = landscape(controlODE, xavier_weights[:], resolution, simplex)
	# zmap = landscape(controlODE, result.params[:], resolution, linear2x)
	# zmap = landscape(controlODE, result.params[:], resolution, barycentric)
	zmap = landscape(controlODE, result.params[:], resolution, simplex)
end

# ╔═╡ b219de76-d25b-4da3-8734-fc2b84335671
@time begin
	plt.clf()
	mshow = plt.matshow(zmap)
	plt.colorbar(mshow)
	plt.gcf()
end

# ╔═╡ 9a261dfc-5844-4b52-b6cf-c4ceb383cd4e
@time begin
	plt.clf()
	cont = plt.contourf(
		resolution,
		resolution,
		zmap;
		# locator=matplotlib.ticker.AutoLocator(),
		extend="both",
		cmap=ColorMap("viridis"),
	)
	plt.xlabel("α")
	plt.ylabel("β")
	plt.colorbar(cont)
	plt.gcf()
end

# ╔═╡ Cell order:
# ╠═31b5c73e-9641-11ec-2b0b-cbd62716cc97
# ╠═07b1f884-6179-483a-8a0b-1771da59799f
# ╠═3d3ef220-af04-4c72-aabe-58d29ae9eb2b
# ╟─ee9852e9-b7a9-4573-a385-45e80f5db1f4
# ╠═4fe5aa44-e7b6-409a-94b5-ae82420e2f69
# ╠═6aa3bbbe-1b38-425d-9e4e-be173f4ddfdd
# ╠═321c438f-0e82-4e6a-a6d3-ff119d6bf556
# ╠═edd99d8c-979c-44a4-bac0-1da3412d4bb4
# ╠═ccba9574-4a9b-4d41-923d-6897482339db
# ╟─2c3e0711-a456-4962-b53b-12d0654704f1
# ╠═87503749-df13-40b3-af30-84da22ddf276
# ╠═4d2ea03b-2c57-424e-b8d3-23950904bf8b
# ╠═65378422-0dd7-4e62-b465-7c1dff882784
# ╠═1db842a6-5d25-451c-9fab-29b9babbe1bb
# ╠═cf18c79a-79b1-4e89-900a-c913d55a7f65
# ╟─13c38797-bc05-4794-b99b-a4aafb2e0503
# ╟─c6557e55-f427-4fee-a04c-b18b0e7b8226
# ╟─5e94cf41-6b08-4a8c-9487-9029b0ad70ce
# ╟─33318735-3495-4ae2-a447-3431aca6e557
# ╟─57628f90-d581-4d2b-82dc-f0a9c300d086
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
# ╟─7559fb6d-b98b-45bb-af12-d088fd74f18c
# ╟─dec21a44-f25d-4a56-bb3e-8b24f913626b
# ╠═08ecf45f-bd51-46c4-94f0-a438a0397c64
# ╟─ae6dbe7f-32dd-438c-be54-ee4d2abee7fd
# ╠═af9cd30d-a8b8-4ef1-83aa-df7aa9036940
# ╠═54b8fb32-0a6d-4a1b-a40e-d0ffd1eedbf0
# ╠═9e3c1ec0-6fa8-4842-98f6-44f26e0dc76b
# ╠═df32b1d0-3d65-4797-9715-500a6e70e2b6
# ╠═2526091c-a56f-4e0b-90f4-79ffa634c436
# ╠═f8716793-b97d-4058-b5a6-8e68d00313b9
# ╠═b1ae1940-6b4a-4a10-bb46-2e692c85c2d3
# ╠═b219de76-d25b-4da3-8734-fc2b84335671
# ╠═9a261dfc-5844-4b52-b6cf-c4ceb383cd4e
