using PyCall

@pyimport tensorflow as tf
@pyimport tensorflow.models.rnn.rnn_cell as tfrnncell
import TensorFlowBuilder: TFParser

target_path = ARGS[1]
const outdir = joinpath(target_path, "src", "API")

println("Generating source code to $outdir")

if !ispath(outdir)
  println("mkdir $outdir")
  mkdir(outdir)
end

@pyimport tensorflow as tf
for (pyimp, pymod, pyname, jlname) in [
                                       (:(tensorflow), :tf, :tf, :Tf),
                                       (:(tensorflow.python.ops.nn), :tf_nn, :(tf.nn), :TfNn),
                                       (:(tensorflow.python.training.training), :tf_train, :(tf.train), :TfTrain),
                                       (:(tensorflow.models.rnn.rnn_cell), :tf_rnn_cell, :(tfrnncell), :TfRnnCell),
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
"Generated automatically by TensorFlowBuilder, from TensorFlow Python version $(version)"
#"TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.""
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
end

println("Loading generated module")

include(joinpath("..", target_path, "src", "TensorFlow.jl"))
using .TensorFlow
