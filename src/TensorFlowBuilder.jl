module TensorFlowBuilder


include("CoreTypes.jl")
using .CoreTypes

include("PyInspector.jl")
using .PyInspector


include("TFParser.jl")
using .TFParser


end # module
