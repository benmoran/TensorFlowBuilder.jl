module CoreTypes
export DimsType, MaybeAbstractTensor

import PyCall: PyObject, @pyimport
@pyimport tensorflow as tf

typealias DimsType Union{Tuple{Vararg{Int64}}, Vector{Int64}}

include("types.jl")
include("dtypes.jl")

end
