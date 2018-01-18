import tensorflow as tf
from tensorflow.contrib.learn import learn_runner
from src.run.network import MyNetwork
from src.lib.fileops import get_all_files_containing
from src.tfrecord.parsers import _img_img_read_and_decode
tf.logging.set_verbosity(tf.logging.DEBUG)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='./models/chpkt/',
    docstring='Output directory for model and training stats, will be under subfolder of model_type.')

tf.app.flags.DEFINE_string(
    flag_name='tfrecord_dir', default_value='/content/tfrecords',
    docstring='Directory where all the tfrecord files are stored')

tf.app.flags.DEFINE_string(
    flag_name='model_type', default_value='simple',
    docstring='Define the model type to use.')


class MyExperiment(MyNetwork):
    def __init__(self, tf_folder, model_type, params):
        MyNetwork.__init__(self, params, model_type)

        # Get the tf records
        self._train_tf = get_all_files_containing(tf_folder, 'train', 'tfrecords')
        self._eval_tf = get_all_files_containing(tf_folder, 'val', 'tfrecords')

        self._dataset_params = {"batch_size": 6,
                                "buffer_size": 6,
                                "num_parallel_calls": 2,
                                "output_buffer_size": 2}

    def get_train_inputs(self):
        # Copy parameters
        params = {**self._dataset_params}

        # Add
        params["tf_records"] = self._train_tf

        def get_next():
            # Create a filename queue for the TFRecords
            dataset = tf.data.TFRecordDataset(params["tf_records"])

            # Parse the record into tensors.
            dataset = dataset.map(_img_img_read_and_decode, num_parallel_calls=params["num_parallel_calls"]) 
            dataset = dataset.prefetch(params["output_buffer_size"])
            dataset = dataset.shuffle(params["buffer_size"])
            dataset = dataset.repeat()  # Repeat the input indefinitely.
            dataset = dataset.batch(params["batch_size"])
            iterator = dataset.make_one_shot_iterator()
            next_example, next_label = iterator.get_next()

            return next_example, next_label

        return get_next

    def get_test_inputs(self):
        # Copy parameters
        params = {**self._dataset_params}

        # Add
        params["tf_records"] = self._eval_tf

        def get_next():
            # Create a filename queue for the TFRecords
            dataset = tf.data.TFRecordDataset(params["tf_records"])

            # Parse the record into tensors.
            dataset = dataset.map(_img_img_read_and_decode, num_parallel_calls=params["num_parallel_calls"]) 
            dataset = dataset.prefetch(params["output_buffer_size"])
            dataset = dataset.shuffle(params["buffer_size"])
            # dataset = dataset.repeat()  # Repeat the input indefinitely.
            dataset = dataset.batch(params["batch_size"])
            iterator = dataset.make_one_shot_iterator()
            next_example, next_label = iterator.get_next()

            return next_example, next_label

        return get_next


    def run_experiment(self, model_dir):
        """Runs the training experiment."""
        # Set the run_config and the directory to save the model and stats
        #with self._graph.as_default():
        run_config = tf.contrib.learn.RunConfig()
        run_config = run_config.replace(model_dir=model_dir)
        run_config = run_config.replace(save_summary_steps=500)

        learn_runner.run(
            experiment_fn=self.experiment_fn,  # First-class function
            run_config=run_config,  # RunConfig
            schedule="train_and_evaluate",  # What to run
            hparams=self._params  # HParams
        )


    def experiment_fn(self, run_config, params):
        """Create an experiment to train and evaluate the model.
            Args:
                run_config (RunConfig): Configuration for Estimator run.
                params (HParam): Hyperparameters
            Returns:
                (Experiment) Experiment for training the mnist model.
        """

        # You can change a subset of the run_config properties as
        run_config = run_config.replace(save_checkpoints_steps=params.min_eval_frequency)

        # Define the mnist classifier
        estimator = self.get_estimator(run_config, params)

        # Get inputs and potential hooks
        train_input_fn = self.get_train_inputs()
        eval_input_fn = self.get_test_inputs()

        # Define the experiment
        experiment = tf.contrib.learn.Experiment(
            estimator=estimator,  # Estimator
            train_input_fn=train_input_fn,  # First-class function
            eval_input_fn=eval_input_fn,  # First-class function
            train_steps=params.train_steps,  # Minibatch steps
            min_eval_frequency=params.min_eval_frequency,  # Eval frequency
            train_monitors=None,  # Hooks for training
            eval_hooks=None,  # Hooks for evaluation
            eval_steps=None  # Use evaluation feeder until its empty
        )

        return experiment

def main(_):
    
    # Define model parameters
    params = tf.contrib.training.HParams(
        learning_rate=0.002,
        n_classes=10,
        train_steps=15000,
        min_eval_frequency=100
    )

 
    Exp = MyExperiment(FLAGS.tfrecord_dir, FLAGS.model_type, params=params)
    Exp.run_experiment(FLAGS.model_dir)

if __name__ == "__main__":
    tf.app.run()