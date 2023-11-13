module TranscriptionNeuralODEs

using Reexport: @reexport

include("LuxUtils.jl")
include("CustomNeuralNetwork.jl")
include("InfOptUtils.jl")
include("ControlUtils.jl")
include("PlotUtils.jl")
include("LossUtils.jl")

@reexport import .CustomNeuralNetwork: chain
@reexport import .LuxUtils: vector_to_parameters
@reexport import .InfOptUtils: optimize_infopt!, extract_infopt_results
@reexport import .ControlUtils: ControlODE, solve, run_simulation
@reexport import .PlotUtils: ShadeConf, square_bounds, states_markers, phase_portrait
@reexport import .LossUtils: dict_weights_randn, dict_weights, simplex

greet() = print("Hello darkness my old foe!")

end  # module TNODEs
