"""
This file contains code that will kick off training and testing processes
"""
import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from sklearn.model_selection import train_test_split

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"/home/workspace/src/data/files/"
        self.n_epochs = 8 # Had to set at 8, workspace crashes and never gets to 10, wasted many hours trying!
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = "/home/workspace/src/data/predictions/"

if __name__ == "__main__":
    # Get configuration

    c = Config()

    # Load data
    print("Loading data...")

    # LoadHippocampusData
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like
    # a k-fold training and combining the results.

    split = dict()

    # Create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation
    # and testing respectively.
    split['train'], split['test'] = train_test_split(keys, test_size=0.25, random_state=30)
    split['train'], split['val'] = train_test_split(split['train'], test_size=0.25, random_state=30)

    # Set up and run experiment

    # UNetExperiment start
    exp = UNetExperiment(c, split, data)

    # run training
    exp.run()

    # prep and run testing

    # Test method run
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))
