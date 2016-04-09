module API
import Base: source_path
apipath = joinpath(dirname(source_path()), "..", "src","API")
for fname in filter(fn -> ismatch(r"^T.*\.jl$",fn), readdir(apipath))
  include(joinpath("API", fname))
  modname = symbol(splitext(fname)[1])
  @eval using .$modname
end

end
