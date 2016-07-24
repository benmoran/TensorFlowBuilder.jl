import TensorFlowBuilder.TFParser: greparg, grepreturn, TFFunction, findarg
import TensorFlowBuilder.TFParser
import TensorFlowBuilder.PyInspector
import TensorFlowBuilder.CoreTypes: Tensor

using PyCall
@pyimport tensorflow as tf
@pyimport tensorflow.python.ops.rnn_cell as tf_rnn_cell


q = TFParser.PyInspector.pydoc(tf.random_normal)
s = "Outputs random values from a normal distribution.

  Args:
    shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
    mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
      distribution.
    stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
      of the normal distribution.
    dtype: The type of the output.
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
      for behavior.
    name: A name for the operation (optional).

  Returns:
    A tensor of the specified shape filled with random normal values.
  "

@test greparg(s, "shape") == "A 1-D integer Tensor or Python array. The shape of the output tensor."
@test greparg(s, "mean") == "A 0-D Tensor or Python value of type `dtype`. The mean of the normal distribution."


@test greparg(s, "seed") == "A Python integer. Used to create a random seed for the distribution. See [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed) for behavior."

@test grepreturn(s) == "A tensor of the specified shape filled with random normal values."

tff = TFFunction(tf.random_normal)

@test tff.name == :random_normal
@test tff.returndoc == grepreturn(s)
@test length(tff.args) == 6
namearg = tff.args[6]
@test !namearg.mandatory
@test namearg.doc == greparg(s, "name")
@test namearg.doc1 == "A name for the operation (optional)."
shapearg = findarg(tff, :shape)
@test shapearg.doc1 == "A 1-D integer Tensor or Python array"


tff2 = TFFunction(tf.slice)
@test TFParser.jlname(findarg(tff2, :begin).name) == "begin_"

tff3 = TFFunction(tf.constant)
@test TFParser.guesstype(:value, findarg(tff3, :value).doc1, nothing) == Tensor



tff4 = TFFunction(tf_rnn_cell.BasicRNNCell[:__init__])
@test tff4.args[4].typesym == Function
