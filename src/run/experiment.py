import tensorflow as tf

tf.logging.set_verbosity(tf.logging.DEBUG)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='./models/chpkt/mnist',
    docstring='Output directory for model and training stats.')


class MyExperiment():
    def __init__(self, train_tf_files, eval_tf_files, tf_parser):

        # Define model parameters
        params = tf.contrib.training.HParams(
            learning_rate=0.002,
            n_classes=10,
            train_steps=500000,
            min_eval_frequency=1000
        )

        MyNetwork.__init__(self, params)

        # Get the tf records
        self._train_tf = train_tf_files
        self._eval_tf = eval_tf_files

        self._tf_parser = tf_parser
        self._dataset_params = {"batch_size": 600, 
                                "buffer_size": 6, 
                                "num_threads": 2, 
                                "output_buffer_size": 2}


    def get_train_inputs(self):
        # Copy parameters
        params = {**self._dataset_params}

        # Add
        params["tf_records"] = [self._train_tf]

        def get_next():
            # Create a filename queue for the TFRecords
            dataset = tf.contrib.data.TFRecordDataset(params["tf_records"])

            # Parse the record into tensors.
            dataset = dataset.map(self._tf_parser, num_threads=params["num_threads"], output_buffer_size=params["output_buffer_size"]) 
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
        params["tf_records"] = [self._test_tf]

        def get_next():
            # Create a filename queue for the TFRecords
            dataset = tf.contrib.data.TFRecordDataset(params["tf_records"])

            # Parse the record into tensors.
            dataset = dataset.map(self._tf_parser, num_threads=params["num_threads"], output_buffer_size=params["output_buffer_size"]) 
            dataset = dataset.shuffle(params["buffer_size"])
            # dataset = dataset.repeat()  # Repeat the input indefinitely.
            dataset = dataset.batch(params["batch_size"])
            iterator = dataset.make_one_shot_iterator()
            next_example, next_label = iterator.get_next()

            return next_example, next_label

        return get_next


    def run_experiment(self, argv=None):
        """Runs the training experiment."""


        # Set the run_config and the directory to save the model and stats
        #with self._graph.as_default():
        run_config = tf.contrib.learn.RunConfig()
        run_config = run_config.replace(model_dir=FLAGS.model_dir)
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

        #with self._graph.as_default():
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
    Exp = MyExperiment()
    Exp.run_experiment()

if __name__ == "__main__":
    tf.app.run()