import os

if __name__ == "__main__":
    cmd = "python3 -m src.trainer.task"
    cmd = "%s %s" % (cmd, "--data_dir /home/adamf/data/carvana/tfrecords")
    cmd = "%s %s" % (cmd, "--output_dir /home/adamf/data/carvana/models")
    cmd = "%s %s" % (cmd, "--model_type simple")
    cmd = "%s %s" % (cmd, "--batch_size 5")
    os.system(cmd)