### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ f8d2ab16-adc4-4c5b-91e2-e9f00d094456
using MLJ

# ╔═╡ f0d94a2c-cadf-4bb9-96e5-5f587677c6bb
using PrettyPrinting

# ╔═╡ 054ec25d-afd7-4617-8cec-dd77643e46e0
using PythonPlot

# ╔═╡ d09d9840-3e29-4679-9cb9-1f513aef289e
using StableRNGs

# ╔═╡ 0f9db1fd-1219-4f8f-a507-adad47b20cd2
using MLJXGBoostInterface

# ╔═╡ c4e67478-ad35-40b5-bd53-f5cd07944ec7
using DataFrames

# ╔═╡ c9e61079-b449-42f1-9b35-8929e23eef83
using StatsBase

# ╔═╡ 463aefa0-4c2b-4178-a4c9-0c47d10a0403
using XGBoost: XGBoost

# ╔═╡ 39a82e59-0cd9-4251-a839-ac729eac7118
using Term

# ╔═╡ ce7df237-a86a-4cab-afaa-8a14a0330d6d
using SIRUS

# ╔═╡ 056ae5d1-c552-40de-af58-6c381a81c38a
using SymbolicRegression

# ╔═╡ cac2192b-03ec-49cd-9b21-41afd5eb16ca
# Based on https://juliaai.github.io/DataScienceTutorials.jl/end-to-end/boston-lgbm/
# Switched from LGB to XGBoost due to installation issues (maybe https://github.com/IQVIA-ML/LightGBM.jl/issues/122)

# ╔═╡ bce3b7f2-a731-4862-b5a4-cd5684eb0d89
import Statistics

# ╔═╡ b28bfb57-2538-4b46-b7f0-9b017bfcd97b
begin	
	raw_features, raw_targets = @load_ames
	raw_features = DataFrame(raw_features)
	select!(raw_features, names(raw_features, Number))
end

# ╔═╡ a3179a98-334a-4312-ba9f-572da8a53769
train, test = partition(collect(eachindex(raw_targets)), 0.70, shuffle=true,
                        rng=StableRNG(52))

# ╔═╡ 98b7dcdc-5766-4aa5-af6a-53e59fce53d8
begin
	mat = Matrix(raw_features)
	Z_features = fit(ZScoreTransform, mat[train, :]; dims=1)
	StatsBase.transform!(Z_features, mat)
	features = DataFrame(mat, names(raw_features))

	Z_targets = fit(ZScoreTransform, raw_targets[train, :]; dims=1)
	targets = StatsBase.transform(Z_targets, raw_targets)

end

# ╔═╡ f440a296-53ae-45b7-8e89-77ac5895ed0e
size(features)

# ╔═╡ 1b2e4d35-ed89-4038-9498-c2b94f12168a
size(targets)

# ╔═╡ 3cc8eb3e-7fbd-42d4-8763-472ff0e5f5e5
typeof(targets)

# ╔═╡ f07b34f7-f930-4e4d-af04-fc1b79234ecd
function testset_rms_unzscore(predictions)
	rescaled = StatsBase.reconstruct(Z_targets, predictions)
	return rms(rescaled, raw_targets[test])
end

# ╔═╡ 87629234-e378-41f0-8406-ce7e4dd8542d
describe(features)

# ╔═╡ 5ffd0402-8c9e-4db7-95c4-ff1b90248edb


# ╔═╡ 6179f6e7-0264-4b7b-9f44-8f43763f3718
# Let us investigation some of the commonly tweaked LightGBM parameters. We start with looking at a learning curve for number of boostings.

xgboost = XGBoostRegressor() #initialised a model with default params

# ╔═╡ aceafce9-ccdd-4ffd-a419-5058fcf08828
begin
	xgboost_m = machine(xgboost, raw_features[train, :], raw_targets[train, 1])
	fit!(xgboost_m)
end

# ╔═╡ 85d197be-8097-4a90-a5d2-fdc5ade55028
# Great, and now let's predict using the held out data.

xgboost_predictions = MLJ.predict(xgboost_m, raw_features[test, :])

# ╔═╡ 1ceaf70c-273a-4a3e-81ee-46d16da3631c
xgboost_score = rms(xgboost_predictions, raw_targets[test])

# ╔═╡ 23c2a17c-7805-4b90-bb65-5feaec521c1d
begin
	df = DataFrame(XGBoost.importancetable(xgboost_m.fitresult[1]))
	name_lookup = Dict(1:ncol(features) .=> names(features))
	transform!(df, :feature => ByRow(f -> name_lookup[f]) => :feature)
end

# ╔═╡ eeedaca4-b4b8-4b89-bcd4-65ffdfe3aec6
first(df, 10)

# ╔═╡ 4787e4c3-f1db-4ee8-a320-eb72f34e9c21
# Now let us try with SIRUS

# ╔═╡ 0740726d-4ae2-449e-baa1-a9745393ee26
sirus = StableRulesRegressor(; max_rules=20, max_depth=2)

# ╔═╡ 0f3c919d-b0c3-4783-8d5a-6c7135af3f43
begin
	sirus_m = machine(sirus, features[train, :], targets[train, 1])
	fit!(sirus_m)
end

# ╔═╡ 555d2b0c-c50d-4c54-a2a5-42cd86819262
sirus_predictions = MLJ.predict(sirus_m, features[test, :])

# ╔═╡ ff006468-b792-48f9-be58-5ca0b2db6a98
sirus_score = testset_rms_unzscore(sirus_predictions)

# ╔═╡ 8b9d5411-5343-459e-922a-94fedc10ab64
sirus_m.fitresult

# ╔═╡ 2b9c1bdf-cec0-49a7-9e42-f37cffcdf0cc
# Now let us see if we can get similar rules with SymbolicRegression.jl

# ╔═╡ 356c9429-69a7-425f-bd9d-85f221029bde
# Define our own `>` and `<` to output numbers of the same type

# ╔═╡ 1466c68e-4afb-4d44-b6c4-96420652048a
function gt(x,y)
	T = promote_type(typeof(x), typeof(y))
	return T(x>y)
end

# ╔═╡ 5e905294-4013-4067-ad94-6f505b7c9001
function lt(x,y)
	T = promote_type(typeof(x), typeof(y))
	return T(x<y)
end

# ╔═╡ b06fdefc-2f03-4dec-8b1f-c66fa7564977
# Use only `+`, `-`, `lt`, `gt`, which is what SIRUS seems to use.
# (One reason SymbolicRegression is interesting in this context is we could add more operations easily!)

# ╔═╡ 6aa189f6-8c74-4b35-881b-1b7266b76f01
options = SymbolicRegression.Options(
    binary_operators=[+, -, lt, gt, *, /],
	# Made some guesses to try to get better results
    topn=30,
	complexity_of_constants=0.1,
);

# ╔═╡ 05331387-8d87-4424-81c6-d21778c87112
hall_of_fame = equation_search(
    permutedims(Matrix(features[train, :])), targets[train, 1], niterations=100, options=options,
    parallelism=:multithreading,
	variable_names = names(features)
)

# ╔═╡ d3952cc4-bc8d-40d7-a246-f8a37552e9e2
# These tradeoff between complexity and loss
dominating = calculate_pareto_frontier(hall_of_fame)

# ╔═╡ 3039d27e-bebd-4bd6-81d3-9e522dc52b7b
trees = [member.tree for member in dominating]

# ╔═╡ e681f78e-60f7-4cc3-bc38-b0fc2d974741
# Take highest complexity result; still not too complicated
tree = trees[end]

# ╔═╡ c8039783-ffa9-4978-aa76-8deee4a83981
# Predict over test set
symbolic_predictions, did_succeed = eval_tree_array(tree,  permutedims(Matrix(features[test, :])), options)

# ╔═╡ e4744648-9367-42e5-a6d0-4bba27f4cdc3
symbolic_score = testset_rms_unzscore(symbolic_predictions)

# ╔═╡ 3fe89337-4fe3-428b-85f8-50ba5b732f9c
symbolic_score

# ╔═╡ bc0a8c0f-f411-4164-b819-156d9a8c8381
sirus_score

# ╔═╡ 0d4b7945-233d-4714-9fb1-bd79ededdd90
xgboost_score

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJXGBoostInterface = "54119dfa-1dab-4055-a167-80440f4f7a91"
PrettyPrinting = "54e16d92-306c-5ea0-a30b-337be88ac337"
PythonPlot = "274fc56d-3b97-40fa-a1cd-1b4a50311bf9"
SIRUS = "cdeec39e-fb35-4959-aadb-a1dd5dede958"
StableRNGs = "860ef19b-820b-49d6-a774-d7a799459cd3"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
SymbolicRegression = "8254be44-1295-4e6a-a16d-46603ac705cb"
Term = "22787eb5-b846-44ae-b979-8e399b8463ab"
XGBoost = "009559a3-9522-5dbb-924b-0b6ed2b22bb9"

[compat]
DataFrames = "~1.5.0"
MLJ = "~0.19.2"
MLJXGBoostInterface = "~0.3.8"
PrettyPrinting = "~0.4.1"
PythonPlot = "~1.0.2"
SIRUS = "~1.2.1"
StableRNGs = "~1.0.0"
StatsBase = "~0.34.0"
SymbolicRegression = "~0.19.1"
Term = "~2.0.5"
XGBoost = "~2.3.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "7c8427ec64540a2fd28066e8a79caf026b09467e"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "e8c8e0a2be6eb4f56b1672e46004463033daa409"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.4.1"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "8bc0aaec0ca548eb6cf5f0d7d16351650c1ee956"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.2"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "f83ec24f76d4c8f525099b2ac475fc098138ec31"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.4.11"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SnoopPrecompile", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e5f08b5689b1aad068e01751889f2f615c7db36d"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.29"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "89e0654ed8c7aebad6d5ad235d6242c2d737a928"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.3"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "498f45593f6ddc0adff64a9310bb6710e851781b"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.5.0+1"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "5248d9c45712e51e27ba9b30eebec65658c6ce29"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.6.0+0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"
weakdeps = ["JSON", "RecipesBase", "SentinelArrays", "StructTypes"]

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes"]
git-tree-sha1 = "da68989f027dcefa74d44a452c9e36af9730a70d"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.10"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "1cdf290d4feec68824bfb84f4bfc9f3aba185647"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.51.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "d730914ef30a06732bdd9f763f6cc32e92ffbff1"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "4e88377ae7ebeaf29a047aa1ee40826e0b708a5d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.7.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.CondaPkg]]
deps = ["JSON3", "Markdown", "MicroMamba", "Pidfile", "Pkg", "TOML"]
git-tree-sha1 = "741146cf2ced5859faae76a84b541aa9af1a78bb"
uuid = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
version = "0.2.18"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "cf25ccb972fec4e4817764d01c82386ae94f77b4"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.14"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

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

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "e76a3281de2719d7c81ed62c6ea7057380c87b1d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.98"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicExpressions]]
deps = ["Compat", "LinearAlgebra", "LoopVectorization", "MacroTools", "PrecompileTools", "Printf", "Random", "Reexport", "Requires", "TOML", "Zygote"]
git-tree-sha1 = "3888b7dee613f702c89029b5e063671a528ed232"
uuid = "a40a106e-89c9-4ca8-8020-a735e8728b6b"
version = "0.9.0"

    [deps.DynamicExpressions.extensions]
    DynamicExpressionsSymbolicUtilsExt = "SymbolicUtils"

    [deps.DynamicExpressions.weakdeps]
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "0b3b52afd0f87b0a3f5ada0466352d125c9db458"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.2.1"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "c6e4a1fbe73b31a3dea94b1da449503b8830c306"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.21.1"

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
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "2e57b4a4f9cc15e85a24d603256fe08e527f48d1"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.8.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "7f5ef966a02a8fdf3df2ca03108a88447cb3c6f0"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.8"

[[deps.Highlights]]
deps = ["DocStringExtensions", "InteractiveUtils", "REPL"]
git-tree-sha1 = "0341077e8a6b9fc1c2ea5edc1e93a956d2aec0c7"
uuid = "eafb193a-b7ab-5a9e-9068-77385905fa72"
version = "0.5.2"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "d38bd0d9759e3c6cfa19bdccc314eccf8ce596cc"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.15"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "0ec02c648befc2f94156eaef13b0f38106212f3f"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.17"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "d7df9a6fdd82a8cfdfe93a94fcce35515be634da"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.3"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1169632f425f79429f245113b775a0e3d121457c"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "5b62d93f2582b09e469b3099d839c2d2ebf5066d"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.13.1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "7d5788011dd273788146d40eb5b1fbdc199d0296"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "6.0.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "1222116d7313cdefecf3d45a2bc1a89c4e7c9217"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.22+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "825289d43c753c7f1bf9bed334c253e9913997f8"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.9.0"

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "88b8f66b604da079a627b6fb2860d3704a6729a1"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.14"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "62f9b2762cc107667b137af621e951f52e020a0f"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.10.2"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "ArrayInterfaceCore", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "e4eed22d70ac91d7a4bf9e0f6902383061d17105"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.162"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.LossFunctions]]
deps = ["CategoricalArrays", "Markdown", "Statistics"]
git-tree-sha1 = "44a7bfeb7b5eb9386a62b9cccc6e21f406c15bea"
uuid = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
version = "0.10.0"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "LinearAlgebra", "MLJBase", "MLJEnsembles", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "ScientificTypes", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "d26cd777c711c332019b39445823cbb1f6cdb7e5"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.19.2"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LinearAlgebra", "LossFunctions", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "ScientificTypes", "Serialization", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "4cc167b6c0a3ab25d7050e4ac38fe119e97cd1ab"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "0.21.11"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJBase", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatsBase"]
git-tree-sha1 = "95b306ef8108067d26dfde9ff3457d59911cc0d6"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.3.3"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "be6d5c71ab499a59e82d65e00a89ceba8732fcd5"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.5.1"

[[deps.MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "Optim", "Parameters"]
git-tree-sha1 = "c92bf0ea37bf51e1ef0160069c572825819748b8"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.9.2"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "c8b7e632d6754a5e36c0d94a4b466a5ba3a30128"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.8.0"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Combinatorics", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "RelocatableFolders", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "38c3b4af6e52edcd94144c32dc1bea335dbfaec7"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.16.8"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase"]
git-tree-sha1 = "02688098bd77827b64ed8ad747c14f715f98cfc4"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.7.4"

[[deps.MLJXGBoostInterface]]
deps = ["MLJModelInterface", "SparseArrays", "Tables", "XGBoost"]
git-tree-sha1 = "43434984d34e9a6e26531d80e6aa0e8bc373b190"
uuid = "54119dfa-1dab-4055-a167-80440f4f7a91"
version = "0.3.8"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MicroMamba]]
deps = ["Pkg", "Scratch", "micromamba_jll"]
git-tree-sha1 = "6f0e43750a94574c18933e9456b18d4d94a4a671"
uuid = "0b3b1443-0f03-428d-bdfb-f27f9c1191ea"
version = "0.1.13"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MyterialColors]]
git-tree-sha1 = "01d8466fb449436348999d7c6ad740f8f853a579"
uuid = "1c23619d-4212-4747-83aa-717207fae70f"
version = "0.3.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "82d7c9e310fe55aa54996e6f7f94674e2a38fcb4"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.9"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg", "Scratch"]
git-tree-sha1 = "6efb039ae888699d5a74fb593f6f3e10c7193e33"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.3.1"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cae3153c7f6cf3f069a853883fd1919a6e5bab5b"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.9+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "e3a6546c1577bfd701771b477b794a52949e7594"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.6"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "4b2e829ee66d4218e0cef22c0a64ee37cf258c29"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.1"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "22a601b04a154ca38867b991d5017469dc75f2db"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "213579618ec1f42dea7dd637a42785a608b1ea9c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.4"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressBars]]
deps = ["Printf"]
git-tree-sha1 = "9d84c8646109eb8bc7a006d59b157c64d5155c81"
uuid = "49802e3a-d2f1-5c88-81d8-b72133a6f568"
version = "1.5.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.PythonCall]]
deps = ["CondaPkg", "Dates", "Libdl", "MacroTools", "Markdown", "Pkg", "REPL", "Requires", "Serialization", "Tables", "UnsafePointers"]
git-tree-sha1 = "0d15cb32f52654921169b4305dae8f66a0e345dc"
uuid = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
version = "0.9.13"

[[deps.PythonPlot]]
deps = ["Colors", "CondaPkg", "LaTeXStrings", "PythonCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "63a18f73425ce301fef12c3f1fb201b720c05015"
uuid = "274fc56d-3b97-40fa-a1cd-1b4a50311bf9"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SIRUS]]
deps = ["AbstractTrees", "CategoricalArrays", "InlineStrings", "LinearAlgebra", "MLJBase", "MLJLinearModels", "MLJModelInterface", "Random", "Statistics", "Tables"]
git-tree-sha1 = "c71a01046a74d956dd94a997d47ca2acfb2c7c35"
uuid = "cdeec39e-fb35-4959-aadb-a1dd5dede958"
version = "1.2.1"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "4b8586aece42bee682399c4c4aee95446aa5cd19"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.39"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "75ccd10ca65b939dab03b812994e571bf1e3e1da"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.2"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SparseMatricesCSR]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "38677ca58e80b5cad2382e5a1848f93b054ad28d"
uuid = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
version = "0.6.7"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "7beb031cf8145577fbccacd94b8a8f4ce78428d3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random", "Test"]
git-tree-sha1 = "3be7d49667040add7ee151fefaf1f8c04c8c8276"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.0"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "dbde6766fc677423598138a5951269432b0fcc90"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.7"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "Requires", "SnoopPrecompile", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "33040351d2403b84afce74dae2e22d3f5b18edcb"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.4.0"

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

    [deps.StaticArrayInterface.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "75ebe04c5bed70b91614d684259b661c9e6274a4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.0"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymbolicRegression]]
deps = ["Dates", "Distributed", "DynamicExpressions", "JSON3", "LineSearches", "LossFunctions", "Optim", "Pkg", "PrecompileTools", "Printf", "ProgressBars", "Random", "Reexport", "Requires", "SpecialFunctions", "StatsBase", "TOML"]
git-tree-sha1 = "d2d6247ab82365be486fb6dfda9b935d27df55d5"
uuid = "8254be44-1295-4e6a-a16d-46603ac705cb"
version = "0.19.1"

    [deps.SymbolicRegression.extensions]
    SymbolicRegressionSymbolicUtilsExt = "SymbolicUtils"

    [deps.SymbolicRegression.weakdeps]
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"

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
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Term]]
deps = ["AbstractTrees", "CodeTracking", "Dates", "Highlights", "InteractiveUtils", "Logging", "Markdown", "MyterialColors", "OrderedCollections", "Parameters", "PrecompileTools", "ProgressLogging", "REPL", "Tables", "UUIDs", "Unicode", "UnicodeFun"]
git-tree-sha1 = "ffac67f6fbcbb32027d924b93ba91b7633af9220"
uuid = "22787eb5-b846-44ae-b979-8e399b8463ab"
version = "2.0.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.UnsafePointers]]
git-tree-sha1 = "c81331b3b2e60a982be57c046ec91f599ede674a"
uuid = "e17b2a0c-0bdf-430a-bd0c-3a23cae4ff39"
version = "1.0.0"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "b182207d4af54ac64cbc71797765068fdeff475d"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.64"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.XGBoost]]
deps = ["AbstractTrees", "CEnum", "JSON3", "LinearAlgebra", "OrderedCollections", "SparseArrays", "SparseMatricesCSR", "Statistics", "Tables", "XGBoost_jll"]
git-tree-sha1 = "d3374bbb77326d7f79ab1cff91b31b1978b871b0"
uuid = "009559a3-9522-5dbb-924b-0b6ed2b22bb9"
version = "2.3.0"

    [deps.XGBoost.extensions]
    XGBoostCUDAExt = "CUDA"
    XGBoostTermExt = "Term"

    [deps.XGBoost.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Term = "22787eb5-b846-44ae-b979-8e399b8463ab"

[[deps.XGBoost_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "de96d469d5da2f48f2fbf8e5dc26e0f94e7e1a1b"
uuid = "a5c6f535-4255-5ca2-a466-0e519f119c46"
version = "1.7.5+2"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5be3ddb88fc992a7d8ea96c3f10a49a7e98ebc7b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.62"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.micromamba_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "087555b0405ed6adf526cef22b6931606b5af8ac"
uuid = "f8abcde7-e9b7-5caa-b8af-a437887ae8e4"
version = "1.4.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═cac2192b-03ec-49cd-9b21-41afd5eb16ca
# ╠═f8d2ab16-adc4-4c5b-91e2-e9f00d094456
# ╠═f0d94a2c-cadf-4bb9-96e5-5f587677c6bb
# ╠═bce3b7f2-a731-4862-b5a4-cd5684eb0d89
# ╠═054ec25d-afd7-4617-8cec-dd77643e46e0
# ╠═d09d9840-3e29-4679-9cb9-1f513aef289e
# ╠═0f9db1fd-1219-4f8f-a507-adad47b20cd2
# ╠═c4e67478-ad35-40b5-bd53-f5cd07944ec7
# ╠═c9e61079-b449-42f1-9b35-8929e23eef83
# ╠═b28bfb57-2538-4b46-b7f0-9b017bfcd97b
# ╠═a3179a98-334a-4312-ba9f-572da8a53769
# ╠═98b7dcdc-5766-4aa5-af6a-53e59fce53d8
# ╠═f440a296-53ae-45b7-8e89-77ac5895ed0e
# ╠═1b2e4d35-ed89-4038-9498-c2b94f12168a
# ╠═3cc8eb3e-7fbd-42d4-8763-472ff0e5f5e5
# ╠═f07b34f7-f930-4e4d-af04-fc1b79234ecd
# ╠═87629234-e378-41f0-8406-ce7e4dd8542d
# ╠═5ffd0402-8c9e-4db7-95c4-ff1b90248edb
# ╠═6179f6e7-0264-4b7b-9f44-8f43763f3718
# ╠═aceafce9-ccdd-4ffd-a419-5058fcf08828
# ╠═85d197be-8097-4a90-a5d2-fdc5ade55028
# ╠═1ceaf70c-273a-4a3e-81ee-46d16da3631c
# ╠═463aefa0-4c2b-4178-a4c9-0c47d10a0403
# ╠═39a82e59-0cd9-4251-a839-ac729eac7118
# ╠═23c2a17c-7805-4b90-bb65-5feaec521c1d
# ╠═eeedaca4-b4b8-4b89-bcd4-65ffdfe3aec6
# ╠═4787e4c3-f1db-4ee8-a320-eb72f34e9c21
# ╠═ce7df237-a86a-4cab-afaa-8a14a0330d6d
# ╠═0740726d-4ae2-449e-baa1-a9745393ee26
# ╠═0f3c919d-b0c3-4783-8d5a-6c7135af3f43
# ╠═555d2b0c-c50d-4c54-a2a5-42cd86819262
# ╠═ff006468-b792-48f9-be58-5ca0b2db6a98
# ╠═8b9d5411-5343-459e-922a-94fedc10ab64
# ╠═2b9c1bdf-cec0-49a7-9e42-f37cffcdf0cc
# ╠═056ae5d1-c552-40de-af58-6c381a81c38a
# ╠═356c9429-69a7-425f-bd9d-85f221029bde
# ╠═1466c68e-4afb-4d44-b6c4-96420652048a
# ╠═5e905294-4013-4067-ad94-6f505b7c9001
# ╠═b06fdefc-2f03-4dec-8b1f-c66fa7564977
# ╠═6aa189f6-8c74-4b35-881b-1b7266b76f01
# ╠═05331387-8d87-4424-81c6-d21778c87112
# ╠═d3952cc4-bc8d-40d7-a246-f8a37552e9e2
# ╠═3039d27e-bebd-4bd6-81d3-9e522dc52b7b
# ╠═e681f78e-60f7-4cc3-bc38-b0fc2d974741
# ╠═c8039783-ffa9-4978-aa76-8deee4a83981
# ╠═e4744648-9367-42e5-a6d0-4bba27f4cdc3
# ╠═3fe89337-4fe3-428b-85f8-50ba5b732f9c
# ╠═bc0a8c0f-f411-4164-b819-156d9a8c8381
# ╠═0d4b7945-233d-4714-9fb1-bd79ededdd90
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
