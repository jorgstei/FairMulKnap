module FairMulKnap

import Pkg
Pkg.add("Combinatorics")
Pkg.add("BenchmarkTools")
Pkg.add("JuMP")
Pkg.add("Cbc")
Pkg.add("LinearAlgebra")
Pkg.add("Plots")
Pkg.add("Profile")
using Random, Combinatorics, BenchmarkTools, JuMP, Cbc, LinearAlgebra, Plots, Profile

include("types.jl")
include("util.jl")
include("linear_model.jl")
include("algorithms.jl")

include("test.jl")

end # module FairMulKnap
