using PyCall
import Base: convert
import PyCall: PyObject

macro pywrapper(typename, supertype)
  quote
    immutable $typename <: $supertype
      x::PyCall.PyObject
    end
    PyCall.PyObject(o::$typename) = o.x
    isequal(x::$typename, y::$typename) = x.x[__eq__](y.x)
    export $typename
  end
end

abstract SessionRunnable
abstract AbstractTensor <: SessionRunnable

export SessionRunnable, AbstractTensor

@pywrapper Dtype Any

@pywrapper Tensor AbstractTensor
@pywrapper Variable AbstractTensor
@pywrapper SparseTensor AbstractTensor
@pywrapper Placeholder AbstractTensor


@pywrapper Operation SessionRunnable

@pywrapper Session Any

typealias MaybeAbstractTensor Union{Void, AbstractTensor}

convert(::Type{Tensor}, a::Array) = Tensor(a)
convert(::Type{AbstractTensor}, a::Array) = Tensor(a)
