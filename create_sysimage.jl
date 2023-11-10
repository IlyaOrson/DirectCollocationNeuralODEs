# run without startup file
# julia --startup-file=no create_sysimage.jl

using PackageCompiler: create_sysimage
import Pkg

Pkg.activate(".")
# deps = [pair.second for pair in Pkg.dependencies()]
# direct_deps = filter(p -> p.is_direct_dep, deps)
# pkg_name_version = [(x.name, x.version) for x in direct_deps]
# pkg_list = [Symbol(x.name) for x in direct_deps]

create_sysimage(
    # filter(x -> x != :BinaryProvider, pkg_list);
    sysimage_path="tnode.so",
    precompile_execution_file="van_der_pol.jl",
)

# how to use sysimage to execute examples & activate current project:
# julia --sysimage tnode.so --project

# how to load Pluto with sysimage
# julia> import Pluto
# julia> Pluto.run(sysimage="tnode.so")
