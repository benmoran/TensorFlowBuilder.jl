"""
Generate Julia wrapping code with type annotations,
by inspecting TensorFlow Python modules and docs
"""
module TFParser

export tfwritejulia

using PyCall

using ..PyInspector
using ..CoreTypes


@pyimport tensorflow as tf

const RESERVED = union(PyCall.reserved, ["begin","end"])

typealias TypeOrUnion Union{DataType,Union}
immutable TFArg
  name::Symbol
  default::Any
  doc::AbstractString
  doc1::AbstractString
  mandatory::Bool
  typesym::TypeOrUnion
end

"Avoid name collisions between TensorFlow functions and Julia reserved words/Base functions"
jlname(s::AbstractString) = (s in RESERVED || isdefined(symbol(s))) ? "$(s)_" : s
jlname(s::Symbol) = jlname(string(s))


# At code generation time, we can't yet import the TensorFlow package itself
# because it depends on the generated code.  So to get the types we need, we
# will import TensorFlow.CoreTypes as TFParser.CoreTypes.  Then we sometimes need to strip
# off the module prefix from the type names to use in the generated code.  Hence:
"Strips off module qualifications from type names"
relativename(dt::DataType) = rsplit(string(dt), '.', limit=2)[end]
relativename(u::Union) = "Union{$(join(sort(map(relativename, u.types)), ','))}"

"Format a TensorFlow function argument as a Julia function argument"
function jlformat(a::TFArg)
  s = jlname(a.name)
  if a.typesym != nothing
    s = "$(s)::$(relativename(a.typesym))"
  end
  if !a.mandatory
    if haskey(CoreTypes.DTYPES, a.default)
      "$s=$(CoreTypes.DTYPES[a.default])"
    elseif ispyfunc(a.default)
      # replace Python functions with their wrapped Julia versions
      "$s=$(jlname(a.default[:__name__]))"
    else
      "$s=$(repr(a.default))"
    end
  else
    s
  end
end

"Pass Julia arguments into Python functios using a **kwargs dict to deal with reserved names"
function pyformat(a::TFArg)
  if a.typesym == Tensor
    # TODO extend/vary?
    ":$(a.name)=>$(jlname(a.name)).x"
  else
    ":$(a.name)=>$(jlname(a.name))"
  end
end


"Try to infer the type of a Tensorflow function argument from name, docstring and default value"
function guesstype(name::Symbol, doc1::AbstractString, default::Any)
  if ispyfunc(default)
    Function
  elseif name == :dtype
    Dtype
  elseif name == :name
    AbstractString
  elseif name == :shape
    Union{AbstractTensor, DimsType}
  elseif name == :strides
    PyVectorType
  elseif name == :seed
    Int
  elseif name == :keep_dims
    Bool
  elseif name == :placeholder
    Placeholder
  elseif name == :Variable
    Variable
  elseif startswith(string(name), "use_")
    Bool
  elseif startswith(string(name), "num_")
    Int
  elseif endswith(string(name), "_size")
    Int
  else
    guesstype(doc1)
  end
end

function guesstyperet(name::Symbol, doc1::AbstractString)
  if name == :dtype
    Dtype
  elseif name == :shape
    Tensor
  elseif name == :placeholder
    Placeholder
  elseif name == :Variable
    Variable
  elseif name == :Session
    Session
  elseif name == :InteractiveSession
    Session
  elseif endswith(string(name), "Optimizer")
    Optimizer
  else
    guesstype(doc1)
  end
end
guesstyperet(name::Symbol, ::Void) = Any

"Try to infer the type of a Tensorflow function's argument or return value from its docstring"
function guesstype(docstring::AbstractString)
  if ismatch(r"[Tt]ensor", docstring) || ismatch(r"of (output )?type `dtype`", docstring)
    Tensor
  elseif ismatch(r"SparseTensor", docstring)
    SparseTensor
  elseif ismatch(r"d?type", docstring)
    Dtype
  elseif ismatch(r"[Aa]n (Python )?(integer|int|`int`)\b", docstring)
    Int
  elseif ismatch(r"[Aa](n optional) `bool`", docstring)
    Bool
  elseif ismatch(r"(^Boolean|^Whether |`?(True|False)`?)", docstring)
    Bool
  elseif ismatch(r"([Aa]|this) `string`", docstring)
    AbstractString
  elseif ismatch(r"\b([Pp]ath|[Dd]irectory|[Nn]ame)\b`", docstring)
    AbstractString
  elseif ismatch(r"[Aa] `tf.Dtype`", docstring)
    Dtype
  elseif ismatch(r": int,`", docstring)
    Int
  else
    # Give up
    Any
  end
end

"All the data we know about a TensorFlow function"
immutable TFFunction
  o::PyObject
  name::Symbol
  args::Vector{TFArg}
  returndoc::AbstractString
  doc::AbstractString
  typesym::DataType
end

"Return the TFArg of the given name in this TFFunction"
findarg(tff::TFFunction, name::Symbol) = tff.args[findfirst(a->a.name == name, tff.args)]

function showmatch(key::AbstractString, m::RegexMatch, ::AbstractString, leading_spaces::Int)
  join_str = "\n" * " "^(leading_spaces+2)
  return strip(replace(m.captures[1], join_str, ' '))
end

"Look in a TensorFlow docstring to find details about a single argument"
function greparg(docstring::AbstractString, key::AbstractString, num_leading_spaces=4)
  # We match everything non-greedily from the key, indented by num_leading_spaces
  # until we encounter either:
  # - a line with non-whitespace after at most num_leading_spaces
  # - or a line with at most some whitespace

  patt_str = "^ {$(num_leading_spaces)}$(key):(.*?)(?=\\n {0,$(num_leading_spaces)}\\w|^\\s*\\z)"
  patt = Regex(patt_str, "ms")
  m = match(patt, docstring)
  if m == nothing
    error("Couldn't find '$key' in docstring:\n$docstring")
  else
    showmatch(key, m, docstring, num_leading_spaces)
  end
end

"Look in a TensorFlow docstring to find details of the return value"
function grepreturn(docstring::AbstractString)
  m = match(r"  Returns:(.*)"ms, docstring)
  (m == nothing) ? "" : showmatch("Returns", m, docstring, 4)
end
grepreturn(::Void) = nothing


"Construct a TFArg from pieces of TensorFlow docstrings"
TFArg(arg::AbstractString, argdoc::AbstractString, default::Any, mandatory::Bool) = begin
  name = symbol(arg)
  doc1 = split(argdoc, ". ", limit=2)[1]
  typesym = guesstype(name, doc1, default)
  if typesym == Tensor
    typesym = AbstractTensor
  end
  if default == nothing
    typesym = Union{typesym, Void}
  end
  println("guessed type $typesym for $arg ($argdoc)")
  TFArg(name, default, argdoc, doc1, mandatory, typesym)
end



"Construct a TFFunction from a TensorFlow Python function"
TFFunction(f::PyObject, name::Symbol; skipself=false) = begin
  argspec = PyInspector.PyArgSpec(f)
  doc = PyInspector.pydoc(f)
  num_defaults = length(argspec.defaults)
  num_mandatory = length(argspec.args) - num_defaults
  tfargs = TFArg[]
  for (i, arg) in enumerate(argspec.args)
    if skipself && (symbol(arg) == :self) && (i==1)
      continue
    end
    j = i - num_mandatory
    if j<=0
      mandatory = true
      default = nothing
    else
      mandatory = false
      default = argspec.defaults[j]
    end

    argdoc = ""
    try
      argdoc = greparg(doc, arg)
    catch
      println("Failed to find doc for $(f[:__name__]) $(arg)")
    end

    push!(tfargs, TFArg(arg, argdoc, default, mandatory))
  end
  retdoc = grepreturn(doc)


  if doc == nothing
    doc = ""
    retdoc = ""
  else
    retdoc = grepreturn(doc)
  end

  TFFunction(f, name, tfargs, retdoc, doc, guesstyperet(name, retdoc))

end

TFFunction(f::PyObject) = TFFunction(f, symbol(f[:__name__]))

function tfwritejulia(mod::Module, pymodname::Symbol)
  src = IOBuffer()
  tfwritejulia(mod, pymodname, src)
  seekstart(src)
  readall(src)
end

tfwritejulia(pymod::PyObject, pymodname::Symbol, io::IO; withdoc=true) =
  tfwritejulia(pywrap(pymod), pymodname, io, withdoc=withdoc)

function tfwritejulia(mod::Module, pymodname::Symbol, io::IO; withdoc=true)
  tfwritejuliatypes(mod, pymodname, io; withdoc=withdoc)
  tfwritejuliafuncs(mod, pymodname, io; withdoc=withdoc)
end

"A function that takes a TensorFlow module and tries to output Julia code
  for all its top-level functions"
function tfwritejuliafuncs(mod::Module, pymodname::Symbol, io::IO; withdoc=true)
  for f in PyInspector.pyfuncs(mod)
    # we want to guess types for each of the arguments
    # identify wrapped PyTypes and annotate them, extracting their PyObjects in the call
    tff = TFFunction(f)
    jlarglist = join(map(jlformat, tff.args), ", ")
    pyarglist = join(map(pyformat, tff.args), ", ")
    definition = "$(pymodname).$(tff.name)(;Dict($pyarglist)...)"

    if tff.typesym != Any
      definition = "$(relativename(tff.typesym))($(definition))"
    end

    if withdoc
      write(io, "\n\n\"\"\"\n")
      write(io, replace(tff.doc, "\$", "\\\$"))
      write(io, "\"\"\"")
    end
    write(io, """
\n$(jlname(tff.name))($jlarglist) = $definition
export $(jlname(tff.name))
          """)
  end
end

"A function that takes a TensorFlow module and tries to output Julia code
  for all its top-level types"
function tfwritejuliatypes(mod::Module, pymodname::Symbol, io::IO; withdoc=true)
    for t in filter(t->haskey(t, :__init__),  PyInspector.pytypes(mod))
      name = symbol(t[:__name__])
      init = t[:__init__]
      doc = t[:__doc__]
      if name in names(CoreTypes)
        # Don't declare it again.
        # (Could still import it and add a new constructor?)
        continue
      end
      tff = nothing
      try
        tff = TFFunction(init, name; skipself=true) # TODO fix the type instead of guessing
      catch e
        println("Skipping $name")
        showerror(STDERR, e, catch_backtrace())
        continue
      end

    jlarglist = join(map(jlformat, tff.args), ", ")
    pyarglist = join(map(pyformat, tff.args), ", ")
    definition = "$(pymodname).$(tff.name)(;Dict($pyarglist)...)"

    if tff.typesym != Any
      definition = "$(relativename(tff.typesym))($(definition))"
    end

    if withdoc
      write(io, "\n\n\"\"\"\n")
      write(io, replace(tff.doc, "\$", "\\\$"))
      write(io, "\"\"\"")
    end
    write(io, """
\n$(jlname(tff.name))($jlarglist) = $definition
export $(jlname(tff.name))
          """)
  end
end

end # module
