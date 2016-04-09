using PyCall

@pyimport tensorflow as tf
using .TFParser

target_path = ARGS[1]
const outdir = joinpath(target_path, "src", "API")

if !ispath(outdir)
  mkdir(outdir)
end

module API; end

@pyimport tensorflow as tf
for (pyimp, pymod, pyname, jlname) in [
                                       (:(tensorflow), :tf, :tf, :Tf),
                                       (:(tensorflow.python.ops.nn), :tf_nn, :(tf.nn), :TfNn),
                                       (:(tensorflow.python.training.training), :tf_train, :(tf.train), :TfTrain),
                                       ]
  @eval m = $pyname
  fname = joinpath(outdir, "$jlname.jl")
  if ispath(fname)
    println("Deleting $fname")
    rm(fname)
  end
    f = open(fname, "w")
    version = "?"
    try
        version = tf.__version__
    catch
    end
  write(f, """
#"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.""
"Generated automatically by 'make -C TensorFlow/apigen', from TensorFlow Python version $(version)"
module $jlname
using PyCall
@pyimport tensorflow as tf
@pyimport $pyimp as $pymod
import TensorFlow.CoreTypes: *
using TensorFlow.CoreTypes
""")

  TFParser.tfwritejulia(m, pymod, f)
  write(f, "end\n")
  close(f)

  # Load the module to check it's syntactically valid.
  println("Loading $fname")
  eval(API, :(include($fname)))
end
