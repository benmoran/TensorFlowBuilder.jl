module PyInspector
export pyfuncs, ispyfunc, pydoc, PyArgSpec

using PyCall
@pyimport inspect

#const PY_FUNC_TYPE = pyeval("type(lambda:1)")

"True if the object passed is a Python function"
ispyfunc(o) = isa(o, PyObject) && inspect.isroutine(o) # isfunction, ismethod, ismethoddescriptor

"True if the object passed is a Python type"
ispytype(o) = isa(o, PyObject) && inspect.isclass(o)

"Given a module, return the list of its top-level Python functions"
pyfuncs(m::Module) = filter(ispyfunc, [eval(m, n) for n in names(m)])

"Given a module, return the list of top-level Python types"
pytypes(m::Module) = filter(ispytype, [eval(m, n) for n in names(m)])

"Given a module, return the list of __init__ methods of top-level Python types"
pyinits(m::Module) = filter(t->haskey(t, :__init__) && inspect.ismethod(t[:__init__]), pytypes(m))

"Return the docstring for a Python object"
pydoc(o::PyObject) = o[:__doc__]


immutable PyArgSpec
  args::Array{Any}
  varargs::Union{AbstractString, Void}
  varkw::Union{AbstractString, Void}
  defaults::Union{Tuple, Void}
end

PyArgSpec(f::PyObject) = begin
  args, varargs, varkw, defaults = inspect.getargspec(f)
  PyArgSpec(args, varargs, varkw, (defaults == nothing) ? () : defaults)
end


end
