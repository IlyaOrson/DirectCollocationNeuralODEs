module ControlUtils

using ArgCheck: @argcheck, @check
using SciMLBase: AbstractODEAlgorithm, AbstractODEProblem, ODEProblem, DECallback
using DiffEqCallbacks: FunctionCallingCallback
import OrdinaryDiffEq

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

end  # ControlUtils
