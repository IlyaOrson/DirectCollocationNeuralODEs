module InfOptUtils

using ArgCheck: @argcheck
using Lux: Lux
using Suppressor: @capture_out
using InfiniteOpt:
    InfiniteOpt,
    InfiniteModel,
    value,
    has_values,
    supports,
    raw_status,
    termination_status,
    solution_summary,
    optimizer_model

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

end  # JUMPUtils
