import tensorflow as tf
from importlib import import_module

from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner


class MyNetwork():
    def __init__(self, params, arch_type='simple'):
        BasicNetwork.__init__(self, params)
        self._graph = tf.Graph()

        # Import the selected arch
        module = importlib.import_module('src.arch.%s' % arch_type)
        arch = getattr(module, 'model')

        self._arch = arch

    def get_architecture(self, features, mode):
        """Return the output operation following the network architecture.
            
        This function will use an outside function to create the architecture.

        Parameters
        ----------
            features: Tensor
                The input tensor to the network
            mode: Tensorflow string
                Holds which state the network is in, predict, inference etc
            scope: str
                Scope name to but the architecture under
        Returns
        -------
             Logits output Op for the network.
        """

        return self._arch(features, mode)

    def get_loss(self, logits, labels):

        return tf.losses.sparse_softmax_cross_entropy(
                labels=tf.cast(labels, tf.int32),
                logits=logits,
                name='loss_op')

    def get_train_op(self, loss, params, global_step):
        """Get the training Op.
            Args:
                 loss (Tensor): Scalar Tensor that represents the loss function.
                 params (HParams): Hyperparameters (needs to have `learning_rate`)
            Returns:
                Training Op
        """

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def get_eval_metric_ops(self, labels, predictions):
        """Return a dict of the evaluation Ops.
            Args:
                labels (Tensor): Labels tensor for training and evaluation.
                predictions (Tensor): Predictions Tensor.
            Returns:
                Dict of metric results keyed by name.
        """
        #with self._graph.as_default():
        return {
            'Accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                name='accuracy')
        }

    def get_model_fn(self, features, labels, mode, params):
        """Model function used in the estimator.
            Args:
                features (Tensor): Input features to the model.
                labels (Tensor): Labels tensor for training and evaluation.
                mode (ModeKeys): Specifies if training, evaluation or prediction.
                params (HParams): hyperparameters.
            Returns:
                (EstimatorSpec): Model to be run by Estimator.
        """
        is_training = mode == ModeKeys.TRAIN

        # If we are serving, we need to receive dicts
        is_serving = mode == ModeKeys.INFER
        if is_serving:
            features = features["feature"]

        # Define model's architecture
        logits = self.get_architecture(features, mode)

        # Define operations during prediction and evaluations
        if mode in (Modes.PREDICT, Modes.EVAL):
            predicted_pixels = tf.argmax(logits, axis=-1, name='classify_pixels')
            probabilities = tf.nn.softmax(logits, name='softmax_tensor')
      
        # When training and evaluating, we want to calculate loss for summary and backprop
        if mode in (Modes.TRAIN, Modes.EVAL):
            global_step = tf.contrib.framework.get_or_create_global_step()
            loss = self.get_loss(logits, labels)
            tf.summary.scalar('OptimizeLoss', loss)
      
        # If only predict, we will return a json string
        if mode == Modes.PREDICT:
            predictions = {
              'classes': predicted_pixels,
              'probabilities': probabilities
            }
            export_outputs = {
              'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

        # If we are training, we need to get our training operation
        # We have already defined the loss above, so we can use it directly here
        if mode == Modes.TRAIN:
            train_op = self.get_train_op(loss, params, global_step)
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
      
        # In eval mode, we add some metrics, and 
        if mode == Modes.EVAL:
            eval_metric_ops = {
              'accuracy': tf.metrics.accuracy(label_indices, predicted_pixels)
            }
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def get_estimator(self, run_config, params):
        """Return the model as a Tensorflow Estimator object.
            Args:
                 run_config (RunConfig): Configuration for Estimator run.
                 params (HParams): hyperparameters.

        """
        return tf.estimator.Estimator(
                model_fn=self.get_model_fn,  # First-class function
                params=params,  # HParams
                config=run_config  # RunConfig
            )