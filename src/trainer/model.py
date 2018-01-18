import os
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from importlib import import_module

def _cnn_model_fn(features, labels, mode, model_arch, learning_rate):
    """Creates the model function.

    This will handle all the different processes needed when using an Estimator.
    The estimator will change "modes" using the mode flag, and depending on that
    different outputs are provided.

    Parameters
    ----------
    features : Tensor
        4D Tensor where the first dimension is the batch size, then height, width
        and channels
    labels : Tensor
        3D Tensor, where the first dimension is the batch size, then height and width.
        The values here are the class number
    mode : tensorflow.python.estimator.model_fn.ModeKeys
        Class that contains the current mode
    model_arch : method
        The method that will construct the graph, found in src.trainer.arch

    Returns
    -------
    tf.estimator.EstimatorSpec
        The requested estimator spec
    """

    # Logits Layer
    logits = model_arch(features['inputs'], mode)

    # If this is a prediction or evaluation mode, then we return
    # the class probabilities and the guessed pixel class
    if mode in (Modes.PREDICT, Modes.EVAL):
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.PREDICT, Modes.EVAL, Modes.TRAIN):
        predicted_pixels = tf.argmax(input=logits, axis=-1)

    # During training and evaluation, we calculate the loss
    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(softmax)
        tf.summary.scalar('OptimizeLoss', loss)
        tf.summary.image('Feature', features['inputs'])
        tf.summary.image('Labels', tf.expand_dims(tf.cast(labels * 255, tf.uint8), 3))
        pred_estend = tf.cast(tf.expand_dims(predicted_pixels * 255, 3), tf.uint8)
        tf.summary.image('Prediction', pred_estend)

    # When predicting (running inference only, during serving for example) we
    # need to return the output as a dictionary.
    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_pixels,
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    # In training (not evaluation) we perform backprop
    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # If evaluating only, we perform evaluation operations
    if mode == Modes.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_pixels)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def build_estimator(model_dir, model_type,learning_rate):
    """Build the estimator using the desired model type"""
    module = import_module('src.trainer.arch')
    arch = getattr(module, model_type)

    # Wrap the model function such that it returns the desired architecture
    def wrapped_model_fn(features, labels, mode):
        return _cnn_model_fn(features, labels, mode, arch, learning_rate)

    # Add the architecture type to the model directory
    model_dir = os.path.join(model_dir, model_type)

    return tf.estimator.Estimator(model_fn=wrapped_model_fn,
                                  model_dir=model_dir,
                                  config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))

def serving_input_fn():
    """Input function to use when serving the model."""
    # TODO: Add automatic check fo rthe size of the trained images for use here?
    inputs = {'inputs': tf.placeholder(tf.float32, [None, 784])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)