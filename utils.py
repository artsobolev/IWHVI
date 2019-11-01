import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm


def binarize_batch(batch_tuple):
    # We share randomness so that if x == y, they're binarized consistently
    # You won't believe how much training time we saved by switching to this
    # _blazingly fast_ binarization

    shape = batch_tuple[0].shape
    random_bytes = np.random.bytes(np.prod(shape))
    rand = np.frombuffer(random_bytes, dtype=np.uint8).reshape(shape)
    res = [((batch == 255) | (rand < batch)).astype(np.float32) for batch in batch_tuple]
    return res


def batched_run(sess, data, target, get_feed_dict, batch_size, n_repeats=1, tqdm_desc=None):
    tqdm_t = None
    if tqdm_desc is not None:
        tqdm_t = tqdm(total=data.num_examples * n_repeats, unit='sample', desc=tqdm_desc)

    for it in range(n_repeats):
        avg_val = 0
        offset = 0
        for batch in batched_dataset(data, batch_size):
            binarized_batch_x, binarized_batch_y = binarize_batch(batch)
            actual_batch_size = len(binarized_batch_x)
            val = sess.run(target, get_feed_dict(binarized_batch_x, binarized_batch_y))

            offset += actual_batch_size
            avg_val += (val - avg_val) * actual_batch_size / offset

            if tqdm_t is not None:
                tqdm_t.update(actual_batch_size)

        yield avg_val

    if tqdm_t is not None:
        tqdm_t.close()


def print_over(msg):
    field = 'tty_columns'
    if not hasattr(print_over, field):
        try:
            with os.popen('stty size', 'r') as fh:
                _, columns = map(int, fh.read().split())
        except:
            columns = None
        setattr(print_over, field, columns)

    columns = getattr(print_over, field)
    if columns is not None:
        print('\r' + msg + ' ' * (columns - len(msg)))
    else:
        print(msg)


def batched_dataset(data, batch_size):
    batch_size = min(data.num_examples, batch_size)
    total_batches = (data.num_examples - 1) // batch_size + 1
    for _ in range(total_batches):
        yield data.next_batch(batch_size)


def save_weights(sess, save_path, suffix='', saver=None):
    if save_path is None:
        return

    if saver is None:
        saver = tf.train.Saver()

    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, 'model-weights' + ('-' + suffix if suffix else ''))
    saved_file = saver.save(sess, model_path)
    print_over('Saved model to ' + saved_file)


def dynamic_tile_to(x, y):
    if len(x.shape) == len(y.shape):
        return x

    assert len(x.shape) < len(y.shape), 'len(x.shape = {}) > len(y.shape = {})'.format(x.shape, y.shape)
    k = len(x.shape)
    multiples = tf.concat([tf.shape(y)[:-k], [1] * k], axis=0)
    return tf.tile(x[(None,) * len(y.shape[:-k])], multiples)


def tile_to_common_shape(*args):
    largest = max(args, key=lambda x: len(x.shape))
    return [dynamic_tile_to(x, largest) for x in args]


def real_nvp_conditional_template(
        x_cond, hidden_layers, shift_only=False, activation=tf.nn.relu, name=None, *args, **kwargs):
    with tf.name_scope(name, "real_nvp_conditional_template"):

        kernel_initializer = tf.initializers.random_normal(0., 1e-4)

        def _fn(x, output_units, **condition_kwargs):
            """Fully connected MLP parameterized via `real_nvp_template`."""
            if condition_kwargs:
                raise NotImplementedError(
                    "Conditioning not implemented in the default template.")

            x_cond_tiled, x = tile_to_common_shape(x_cond, x)
            for units in hidden_layers:
                x = tf.layers.dense(
                    inputs=tf.concat([x_cond_tiled, x], axis=-1),
                    units=units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    *args,
                    **kwargs
                )

            x = tf.layers.dense(
                inputs=tf.concat([x_cond_tiled, x], axis=-1),
                units=(1 if shift_only else 2) * output_units,
                kernel_initializer=kernel_initializer,
                activation=None,
                *args,
                **kwargs
            )
            if shift_only:
                return x, None
            shift, log_scale = tf.split(x, 2, axis=-1)
            return shift, log_scale

        return tf.make_template("real_nvp_conditional_template", _fn)


def masked_autoregressive_conditional_template(
        x_cond, hidden_layers, shift_only=False, activation=tf.nn.relu,
        log_scale_min_clip=-5., log_scale_max_clip=3., log_scale_clip_gradient=False,
        name=None, *args, **kwargs):
    """Build the Masked Autoregressive Density Estimator (Germain et al., 2015).

    This will be wrapped in a make_template to ensure the variables are only
    created once. It takes the input and returns the `loc` ("mu" in [Germain et
    al. (2015)][1]) and `log_scale` ("alpha" in [Germain et al. (2015)][1]) from
    the MADE network.

    Warning: This function uses `masked_dense` to create randomly initialized
    `tf.Variables`. It is presumed that these will be fit, just as you would any
    other neural architecture which uses `tf.layers.dense`.

    #### About Hidden Layers

    Each element of `hidden_layers` should be greater than the `input_depth`
    (i.e., `input_depth = tf.shape(input)[-1]` where `input` is the input to the
    neural network). This is necessary to ensure the autoregressivity property.

    #### About Clipping

    This function also optionally clips the `log_scale` (but possibly not its
    gradient). This is useful because if `log_scale` is too small/large it might
    underflow/overflow making it impossible for the `MaskedAutoregressiveFlow`
    bijector to implement a bijection. Additionally, the `log_scale_clip_gradient`
    `bool` indicates whether the gradient should also be clipped. The default does
    not clip the gradient; this is useful because it still provides gradient
    information (for fitting) yet solves the numerical stability problem. I.e.,
    `log_scale_clip_gradient = False` means
    `grad[exp(clip(x))] = grad[x] exp(clip(x))` rather than the usual
    `grad[clip(x)] exp(clip(x))`.

    Args:
    x_cond: Tensor to condition on
    hidden_layers: Python `list`-like of non-negative integer, scalars
      indicating the number of units in each hidden layer. Default: `[512, 512].
    shift_only: Python `bool` indicating if only the `shift` term shall be
      computed. Default: `False`.
    activation: Activation function (callable). Explicitly setting to `None`
      implies a linear activation.
    log_scale_min_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The minimum value to clip by. Default: -5.
    log_scale_max_clip: `float`-like scalar `Tensor`, or a `Tensor` with the
      same shape as `log_scale`. The maximum value to clip by. Default: 3.
    log_scale_clip_gradient: Python `bool` indicating that the gradient of
      `tf.clip_by_value` should be preserved. Default: `False`.
    name: A name for ops managed by this function. Default:
      "masked_autoregressive_default_template".
    *args: `tf.layers.dense` arguments.
    **kwargs: `tf.layers.dense` keyword arguments.

    Returns:
    shift: `Float`-like `Tensor` of shift terms (the "mu" in
      [Germain et al.  (2015)][1]).
    log_scale: `Float`-like `Tensor` of log(scale) terms (the "alpha" in
      [Germain et al. (2015)][1]).

    Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.

    #### References

    [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
       Masked Autoencoder for Distribution Estimation. In _International
       Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
    """

    name = name or "masked_autoregressive_conditional_template"
    with tf.name_scope(name, values=[x_cond, log_scale_min_clip, log_scale_max_clip]):
        def _fn(x):
            input_depth = x.shape.with_rank_at_least(1)[-1].value
            if input_depth is None:
                raise NotImplementedError("Rightmost dimension must be known prior to graph execution.")

            input_shape = np.int32(x.shape.as_list()) if x.shape.is_fully_defined() else tf.shape(x)

            for i, units in enumerate(hidden_layers):
                x = tfp.bijectors.masked_dense(
                    inputs=x, units=units, num_blocks=input_depth,
                    exclusive=True if i == 0 else False, activation=activation,
                    *args, **kwargs)
                x += tf.layers.dense(inputs=x_cond, units=units, activation=activation)

            x = tfp.bijectors.masked_dense(
                inputs=x, units=(1 if shift_only else 2) * input_depth,
                num_blocks=input_depth, activation=None, *args, **kwargs)
            x += tf.layers.dense(inputs=x_cond, units=(1 if shift_only else 2) * input_depth, activation=None)

            if shift_only:
                x = tf.reshape(x, shape=input_shape)
                return x, None

            x = tf.reshape(x, shape=tf.concat([input_shape, [2]], axis=0))
            shift, log_scale = tf.unstack(x, num=2, axis=-1)
            which_clip = tf.clip_by_value if log_scale_clip_gradient else _clip_by_value_preserve_grad
            log_scale = which_clip(log_scale, log_scale_min_clip, log_scale_max_clip)
            return shift, log_scale

        return tf.make_template(name, _fn)


def _clip_by_value_preserve_grad(x, clip_value_min, clip_value_max, name=None):
    with tf.name_scope(name, "clip_by_value_preserve_grad",
                       [x, clip_value_min, clip_value_max]):
        clip_x = tf.clip_by_value(x, clip_value_min, clip_value_max)
        return x + tf.stop_gradient(clip_x - x)


class StepSchedule:
    def __init__(self, spec, default):
        self._default = default
        self._schedule = sorted(spec)

        assert all(len(x) == 2 for x in self._schedule), 'Malformed schedule: {}'.format(self._schedule)

    def at(self, cur_step):
        ret = self._default
        for step, val in self._schedule:
            if step > cur_step:
                break
            ret = val

        return ret


class CyclicTemperatureSchedule:
    def __init__(self, total_epochs, cycles, annealing_fraction):
        self._cycle_size = total_epochs // cycles if cycles > 0 else total_epochs
        self._annealing_fraction = annealing_fraction if cycles > 0 else 0.0

    def at(self, cur_step):
        t = (cur_step % self._cycle_size) / self._cycle_size
        return t / self._annealing_fraction if t < self._annealing_fraction else 1.


def make_typed_tuple(*types, rest=None):
    def impl(x):
        x_vals = x.split(',')
        if rest is None:
            assert len(x_vals) == len(types), 'Wrong argument: "{}", expected {} values'.format(x, len(types))
        else:
            assert len(x_vals) >= len(types), 'Wrong argument: "{}", expected at least {} values'.format(x, len(types))

        return tuple((types[idx] if idx < len(types) else rest)(x_val) for idx, x_val in enumerate(x_vals))

    return impl
