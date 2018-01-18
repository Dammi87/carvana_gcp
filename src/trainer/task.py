
import argparse
import os

import src.trainer.model as model
import src.trainer.input as input_pipe

from tensorflow.contrib.learn import Experiment
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)

def generate_experiment_fn(model_type,
                           data_dir,
                           train_steps,
                           eval_steps,
                           learning_rate,
                           input_pipe_settings,
                           **other_experiment_args):
    """Create a method that constructs an experiment.

    Parameters
    ----------
    settings : dict
        A dictionary containing the desired experiment settings
    input_pipe_settings: dict
        A dictionary for settings of the input pipe.
        Note: When these are used for the eval input_fn, epochs are set to 1
            'batch_size' : int
                Batch size to return
            'epochs' : int
                How many epochs to return, None means infinite
            'num_parallel_calls' : int
                How many threads to use

    other_experiment_args:
        Adds possibility to affect other arguments of the experiment

    Returns
    -------
    Method
        Returns the method that returns the desired Experiment to a estimator
    """
    def _experiment_fn(output_dir):

        return Experiment(
            model.build_estimator(output_dir, model_type, learning_rate),
            train_input_fn=input_pipe.get_input_fn("train", data_dir, **input_pipe_settings),
            eval_input_fn=input_pipe.get_input_fn("eval", data_dir, **input_pipe_settings),
            train_steps=train_steps,
            eval_steps=eval_steps,
            export_strategies=[saved_model_export_utils.make_export_strategy(
                model.serving_input_fn,
                default_output_alternative_key=None,
                exports_to_keep=1)],
            **other_experiment_args)

    return _experiment_fn


if __name__ == '__main__':
    # To see available arguments:
    # https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='GCS or local path to training data',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--model_type',
        help='Set which model type to use, see available in src.trainer.arch',
        required=True
    )
    parser.add_argument(
        '--batch_size',
        help='Batch size to use',
        type=int,
        default=10
    )
    parser.add_argument(
        '--train_steps',
        help='Steps to run the training job for.',
        type=int,
        default=10000
    )
    parser.add_argument(
        '--eval_steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=100,
        type=int
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    parser.add_argument(
        '--eval_delay_secs',
        help='How long to wait before running first evaluation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--min_eval_frequency',
        help='Minimum number of training steps between evaluations',
        default=100,
        type=int
    )
    parser.add_argument(
        '--learning_rate',
        help='Learning rate to use',
        default=0.0001,
        type=float
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    output_dir = arguments.pop('output_dir')

    input_pipe_settings = {
        'batch_size': arguments.pop('batch_size'),
        'epochs': None,
        'num_parallel_calls': 2,
        'img_sizes': input_pipe.get_tf_record_image_size(arguments['data_dir'])
    }
    arguments["input_pipe_settings"] = input_pipe_settings

    # Run the training job
    learn_runner.run(generate_experiment_fn(**arguments), output_dir)
