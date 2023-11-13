module PlotUtils

using Base: @kwdef
using ArgCheck: @argcheck
using LazyGrids: ndgrid
using PyPlot: plt

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

end  # PlotUtils module
