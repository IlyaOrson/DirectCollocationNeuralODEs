### A Pluto.jl notebook ###
# v0.19.36

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
	using PlutoUI: TableOfContents
	using Base: @kwdef
	using Random: Random
	using Lux: Lux, Dense, Chain
	using ComponentArrays: ComponentArray, getaxes
	using ProgressLogging: @progress
	using LinearAlgebra: norm

	using Ipopt: Ipopt
    using InfiniteOpt:
        JuMP,
		MOI,
        InfiniteModel,
        OrthogonalCollocation,
        Infinite,
        @infinite_parameter,
        @variable,
        @variables,
        @constraint,
        @constraints,
        @objective,
        ∂,
		integral,
        optimizer_with_attributes,
        optimizer_model,
		solve_time,
		value,
		objective_value,
		all_variables,
		all_parameters,
		all_constraints,
		supports,
		AffExpr

    # using MadNLP: MadNLP

	using ReverseDiff: ReverseDiff
    using ForwardDiff: ForwardDiff
    using Enzyme: Enzyme

	using QuadGK: quadgk
	
	using DataInterpolations: ConstantInterpolation, LinearInterpolation, QuadraticInterpolation, LagrangeInterpolation
	
	using BenchmarkTools: @benchmark

	using ArgCheck: @argcheck, @check

	using UnicodePlots: histogram
	using PyPlot: matplotlib, plt, ColorMap

	using Revise
end

# ╔═╡ 107c170c-32d4-4613-8565-fb881307c1b7
@time @time_imports using TranscriptionNeuralODEs:
	array_to_namedtuple,
	optimize_infopt!,
	extract_infopt_results,
	custom_chain,
	ControlODE,
	solve,
	run_simulation,
	ShadeConf,
	square_bounds,
	states_markers,
	phase_portrait,
	dict_weights_randn,
	dict_weights,
	simplex,
	loss_landscape

# ╔═╡ 42a1e8d7-ad92-444a-bb9d-027ed8a06e22
begin
	import TranscriptionNeuralODEs
	names(TranscriptionNeuralODEs)
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
	layer_size = 16
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

# ╔═╡ 65378422-0dd7-4e62-b465-7c1dff882784
begin
	# Construct the layer
	final_activation(x) = (tanh.(x) .* 2) .- 1.0
	# NOTE avoid automatic convertion from tanh to fast_tanh
	lchain = Chain(
		Dense(2, layer_size, tanh; init_weight=Lux.glorot_normal, allow_fast_activation=false),
		Dense(layer_size, 1, final_activation; init_weight=Lux.glorot_normal)
	)
end;

# ╔═╡ 1db842a6-5d25-451c-9fab-29b9babbe1bb
begin
	# Seeding
	rng = Random.default_rng()
	Random.seed!(rng, 0)

	# Parameter and State Variables
	ps, st = Lux.setup(rng, lchain)

	xavier_weights = ComponentArray(ps)

	# Run the model
	# y, st = Lux.apply(lchain, u0, ps, st)
end;

# ╔═╡ cf18c79a-79b1-4e89-900a-c913d55a7f65
xavier_weights |> histogram

# ╔═╡ 13c38797-bc05-4794-b99b-a4aafb2e0503
md"## Custom feedforward NN"

# ╔═╡ 253458ec-12d1-4228-b57a-73c02b3b2c49
begin
    layer_sizes = ((layer_size for _ in 1:layer_num)..., 1)
    activations = ((tanh for _ in 1:layer_num)..., final_activation)

	# use Lux initialization instead
    # xavier_weights = xavier_sampler(nstates, layer_sizes)
end;

# ╔═╡ 0ac534b3-40b4-44ef-a903-0ea69db883e2
md"## User defined functions API in JuMP"

# ╔═╡ 62945ff7-e39c-40c5-9be7-3e43a1930565
begin
	nstates = length(u0)
	nparams = length(xavier_weights)

	cat_params = vcat(u0 , collect(xavier_weights))
	grad_container =  similar(cat_params)
end

# ╔═╡ 02377f26-039d-4685-ba94-1574b3b18aa6
function vector_fun(z)
	# @show z typeof(z) z[1:nstates] z[nstates+1:end]
	x = collect(z[1:nstates])
	p = collect(z[(nstates + 1):end])
	# return custom_chain(x, p, layer_sizes, activations)  # custom nn
	return Lux.apply(lchain, x, array_to_namedtuple(p, ps), st)[begin]  # lux nn
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

# ╔═╡ 84027727-3198-4884-b71c-ab4ca86ae5d7
xavier_weights == ComponentArray(array_to_namedtuple(xavier_weights, ps))

# ╔═╡ 62626cd0-c55b-4c29-94c2-9d736f335349
md"# Collocation with policy"

# ╔═╡ a86ee119-6f73-4ba5-a255-b13e0bc6ca95
function scope_test(model, x)
	@constraint(model, -0.4 <= x[1])
end

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

    # if constrain_states
    #     @constraint(model, -0.4 <= x[1])  # FIXME
    # end
	scope_test(model, x)

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

# ╔═╡ 732b8e45-fb51-454b-81d2-2d084c12df73
begin
	neural_model = build_neural_collocation(
			num_supports=neural_num_supports,
	    	nodes_per_element=neural_nodes_per_element,
	)
	optimize_infopt!(neural_model; verbose=false)
end

# ╔═╡ edd395e2-58b7-41af-85ae-6af612154df5
result = extract_infopt_results(neural_model);

# ╔═╡ 0ac3a6d6-9354-4944-86a8-205bddc99019
result.params |> histogram

# ╔═╡ c31f52de-34a0-403f-9683-cc38a2386b62
md"## Van Der Pol system in the interface"

# ╔═╡ 281a5a4d-e156-4e04-a013-d1b1351dd822
begin
	@kwdef struct VanDerPol
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
	controller = (x, p) -> Lux.apply(lchain, x, array_to_namedtuple(p, ps), st)[begin]
	controlODE = ControlODE(controller, system, u0, tspan)
end;

# ╔═╡ 2590217d-f9e4-4ddf-a0f8-4830b870fad5
map( # same chain results
	≈,
	Lux.apply(lchain, controlODE.u0, array_to_namedtuple(xavier_weights, ps), st)[1],
	custom_chain(controlODE.u0, xavier_weights, layer_sizes, activations)
) |> all

# ╔═╡ d2f60a56-3615-459b-bbbf-9dee822a7213
map( # same derivatives
	≈,
	ReverseDiff.gradient(p -> custom_chain(controlODE.u0, p, layer_sizes, activations), ComponentArray(ps)),
	ReverseDiff.gradient(p -> sum(Lux.apply(lchain, u0, p, st)[1]), ComponentArray(ps))
) |> all

# ╔═╡ e42a34d5-cef0-4630-a19f-cee50b7851a7
@benchmark Lux.apply($lchain, $controlODE.u0, $xavier_weights, $st)

# ╔═╡ 3b23856e-3ebe-4441-a47f-e9078c824d58
@benchmark custom_chain($controlODE.u0, $xavier_weights, $layer_sizes, $activations)

# ╔═╡ c6603841-f09c-49ac-9b84-81295b09b22b
md"## Discrete and continuous losses"

# ╔═╡ 52afbd53-5128-4482-b929-2c71398be122
function loss_discrete(controlODE, params; inf_penalty=true, kwargs...)
    objective = zero(eltype(params))
    sol = solve(controlODE, params; kwargs...)
	sol_arr = Array(sol)
	if inf_penalty && state_bound_violation(controlODE.system, sol_arr)
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

# ╔═╡ 42f26a4c-ac76-4212-80f9-82858ce2959c
loss_continuous(controlODE, result.params)

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


# ╔═╡ 81acdc27-2a9d-4062-b1e7-d139a3abe5a9
function interpolant_controller(collocation; interpolator=LinearInterpolation)
	# not all interpolators support multidimensional data out-of-the-box
	# but this can be done manually for the time being
	# https://github.com/SciML/DataInterpolations.jl/issues/206

	# TODO generalize this to intepolate states states as well
	num_controls = size(collocation.controls, 1)

    interpolations = [
        interpolator(collocation.controls[i, :], collocation.times) for i in 1:num_controls
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
	times_cp, states_cp, controls_cp = run_simulation(collocationODE, nothing; control_input=:time, dt=1f-2)
end

# ╔═╡ e1c0f4a5-6660-4bd2-bed8-d82665c34f19
@time begin
	plt.clf()
	plt.plot(times_c, controls_c[1,:], label="Collocation control")
	plt.plot(times_cp, controls_cp[1,:], label="Neural control")
	plt.title("Control Profile")
	plt.legend()
	plt.gcf()
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

# ╔═╡ f8716793-b97d-4058-b5a6-8e68d00313b9
# central_point = array_to_namedtuple(xavier_weights[:], ps)
central_point = array_to_namedtuple(result.params[:], ps)

# ╔═╡ b1ae1940-6b4a-4a10-bb46-2e692c85c2d3
@time begin
	resolution = -5f0:1f-1:5f0
	X,Y,zmap = loss_landscape(controlODE, loss_discrete, central_point, resolution, simplex)
end

# ╔═╡ 9a261dfc-5844-4b52-b6cf-c4ceb383cd4e
@time begin
	plt.clf()
	cont = plt.contourf(
		X,
		Y,
		zmap;
		# locator=matplotlib.ticker.AutoLocator(),
		# locator=matplotlib.ticker.MaxNLocator(),
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
# ╠═107c170c-32d4-4613-8565-fb881307c1b7
# ╠═42a1e8d7-ad92-444a-bb9d-027ed8a06e22
# ╠═3d3ef220-af04-4c72-aabe-58d29ae9eb2b
# ╟─ee9852e9-b7a9-4573-a385-45e80f5db1f4
# ╠═4fe5aa44-e7b6-409a-94b5-ae82420e2f69
# ╠═6aa3bbbe-1b38-425d-9e4e-be173f4ddfdd
# ╠═321c438f-0e82-4e6a-a6d3-ff119d6bf556
# ╠═edd99d8c-979c-44a4-bac0-1da3412d4bb4
# ╠═ccba9574-4a9b-4d41-923d-6897482339db
# ╟─2c3e0711-a456-4962-b53b-12d0654704f1
# ╠═65378422-0dd7-4e62-b465-7c1dff882784
# ╠═1db842a6-5d25-451c-9fab-29b9babbe1bb
# ╠═cf18c79a-79b1-4e89-900a-c913d55a7f65
# ╟─13c38797-bc05-4794-b99b-a4aafb2e0503
# ╠═253458ec-12d1-4228-b57a-73c02b3b2c49
# ╟─0ac534b3-40b4-44ef-a903-0ea69db883e2
# ╠═62945ff7-e39c-40c5-9be7-3e43a1930565
# ╠═02377f26-039d-4685-ba94-1574b3b18aa6
# ╠═aa575018-f57d-4180-9663-44da68d6c77c
# ╟─12b71a53-cbe3-4cb3-a383-c341048616e8
# ╠═ceef04c1-7c0b-4e6f-ad2c-3679e6ed0055
# ╠═cf9abc47-962f-4837-a4e5-7e5984dae474
# ╠═09ef9cc4-2564-44fe-be1e-ce75ad189875
# ╟─826d4479-9db0-44d6-98ae-b46e7a6a0b4e
# ╠═84027727-3198-4884-b71c-ab4ca86ae5d7
# ╠═2590217d-f9e4-4ddf-a0f8-4830b870fad5
# ╠═d2f60a56-3615-459b-bbbf-9dee822a7213
# ╠═e42a34d5-cef0-4630-a19f-cee50b7851a7
# ╠═3b23856e-3ebe-4441-a47f-e9078c824d58
# ╟─62626cd0-c55b-4c29-94c2-9d736f335349
# ╠═a86ee119-6f73-4ba5-a255-b13e0bc6ca95
# ╠═d8888d92-71df-4c0e-bdc1-1249e3da23d0
# ╠═732b8e45-fb51-454b-81d2-2d084c12df73
# ╠═edd395e2-58b7-41af-85ae-6af612154df5
# ╠═0ac3a6d6-9354-4944-86a8-205bddc99019
# ╟─c31f52de-34a0-403f-9683-cc38a2386b62
# ╠═281a5a4d-e156-4e04-a013-d1b1351dd822
# ╠═58b96a16-f288-4e34-99b7-e0ea11a347e8
# ╠═3061a099-3d0d-472d-a1b5-e4785b980014
# ╠═1772d71a-1f7f-43cd-a4ad-0f7f54c960d0
# ╟─c6603841-f09c-49ac-9b84-81295b09b22b
# ╠═52afbd53-5128-4482-b929-2c71398be122
# ╠═6704374c-70de-4e4d-9523-e516c1072348
# ╠═d964e018-1e22-44a0-baef-a18ed5979a4c
# ╠═42f26a4c-ac76-4212-80f9-82858ce2959c
# ╠═022af0de-c125-4dd4-82e2-79409e5df76c
# ╟─10529d5d-486e-4455-9876-5ac46768ce8a
# ╠═7eec23a8-1840-47db-aba8-89f9f057d9a4
# ╠═1a9bcfe7-23f5-4c89-946c-0a97194778f0
# ╠═81acdc27-2a9d-4062-b1e7-d139a3abe5a9
# ╠═57ec210a-0dda-4a1c-9478-736d669b7090
# ╠═751dadcb-9cc7-4719-94f0-33455bdf493d
# ╠═e1c0f4a5-6660-4bd2-bed8-d82665c34f19
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
# ╠═f8716793-b97d-4058-b5a6-8e68d00313b9
# ╠═b1ae1940-6b4a-4a10-bb46-2e692c85c2d3
# ╠═9a261dfc-5844-4b52-b6cf-c4ceb383cd4e
