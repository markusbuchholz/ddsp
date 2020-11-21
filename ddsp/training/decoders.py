# Copyright 2020 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Library of decoder layers."""

from ddsp.training import nn
import gin
import tensorflow as tf

tfkl = tf.keras.layers


# ------------------ Decoders --------------------------------------------------
class OutputSplitsLayer(nn.DictLayer):
  """Takes in several tensors, gets single tensor back, splits to dictionary."""

  def __init__(self,
               input_keys=None,
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               **kwargs):
    """Constructor.

    Args:
      input_keys: A list of keys to read out of a dictionary passed to call().
        If no input_keys are provided to the constructor, they are inferred from
        the argument names in compute_outputs().
      output_splits: A list of tuples (output_key, n_channels). Output keys are
        extracted from the list and the output tensor from compute_output(), is
        split into a dictionary of tensors, each with its matching n_channels.
      **kwargs: Other tf.keras.layer kwargs, such as name.
    """
    self.output_splits = output_splits
    self.n_out = sum([v[1] for v in output_splits])
    input_keys = input_keys or self.get_argument_names('compute_output')
    output_keys = [v[0] for v in output_splits]
    super().__init__(input_keys=input_keys, output_keys=output_keys, **kwargs)

  def call(self, *inputs, **unused_kwargs):
    """Splits a single output tensor into a dictionary of output tensors."""
    output = self.compute_output(*inputs)
    return nn.split_to_dict(output, self.output_splits)

  def compute_output(self, *inputs):
    """Runs network that takes multiple tensors and outputs a single tensor."""
    raise NotImplementedError


@gin.register
class RnnFcDecoder(OutputSplitsLayer):
  """RNN and FC stacks for f0 and loudness."""

  def __init__(self,
               rnn_channels=512,
               rnn_type='gru',
               ch=512,
               layers_per_stack=3,
               input_keys=('ld_scaled', 'f0_scaled', 'z'),
               output_splits=(('amps', 1), ('harmonic_distribution', 40)),
               **kwargs):
    super().__init__(
        input_keys=input_keys, output_splits=output_splits, **kwargs)
    stack = lambda: nn.FcStack(ch, layers_per_stack)

    # Layers.
    self.input_stacks = [stack() for k in self.input_keys]
    self.rnn = nn.Rnn(rnn_channels, rnn_type)
    self.out_stack = stack()
    self.dense_out = tfkl.Dense(self.n_out)

  def compute_output(self, *inputs):
    # Initial processing.
    inputs = [stack(x) for stack, x in zip(self.input_stacks, inputs)]

    # Run an RNN over the latents.
    x = tf.concat(inputs, axis=-1)
    x = self.rnn(x)
    x = tf.concat(inputs + [x], axis=-1)

    # Final processing.
    x = self.out_stack(x)
    return self.dense_out(x)


