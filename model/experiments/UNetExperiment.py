"""
This module represents a UNet experiment and contains a class that handles
the experiment lifecycle
"""
import os
import time

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_prep.SlicesDataset import SlicesDataset
from utils.utils import log_to_tensorboard
from utils.volume_stats import Dice3d, Jaccard3d
from networks.RecursiveUNet import UNet
from inference.UNetInferenceAgent import UNetInferenceAgent

class UNetExperiment:
    """
    This class implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    The basic life cycle of a UNetExperiment is:

        run():
            for epoch in n_epochs:
                train()
                validate()
        test()
    """
    def __init__(self, config, split, dataset):
        self.n_epochs = config.n_epochs
        self.split = split
        self._time_start = ""
        self._time_end = ""
        self.epoch = 0
        self.name = config.name

        # Create output folders
        dirname = f'{time.strftime("%Y-%m-%d_%H%M", time.gmtime())}_{self.name}'
        self.out_dir = os.path.join(config.test_results_dir, dirname)
        os.makedirs(self.out_dir, exist_ok=True)
        self.out_images_dir = os.path.join(self.out_dir, "images")
        os.makedirs(self.out_images_dir)

        # Create data loaders
        # Note that we are using a 2D version of UNet here, which means that it will expect
        # batches of 2D slices.
        self.train_loader = DataLoader(SlicesDataset(dataset[split["train"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(SlicesDataset(dataset[split["val"]]),
                batch_size=config.batch_size, shuffle=True, num_workers=0)

        # we will access volumes directly for testing
        self.test_data = dataset[split["test"]]

        # Do we have CUDA available?
        if not torch.cuda.is_available():
            print("WARNING: No CUDA device is found. This may take significantly longer!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure our model and other training implements
        # We will use a recursive UNet model from German Cancer Research Center,
        # Division of Medical Image Computing. It is quite complicated and works
        # very well on this task.
        self.model = UNet(num_classes=3)
        self.model.to(self.device)

        # We are using a standard cross-entropy loss since the model output is essentially
        # a tensor with softmax'd prediction of each pixel's probability of belonging
        # to a certain class
        self.loss_function = torch.nn.CrossEntropyLoss()

        # We are using standard SGD method to optimize our weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        # Scheduler helps us update learning rate automatically
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        # Set up Tensorboard. By default it saves data into runs folder. You need to launch
        self.tensorboard_train_writer = SummaryWriter(comment="_train")
        self.tensorboard_val_writer = SummaryWriter(comment="_val")

    def train(self):
        """
        This method is executed once per epoch and takes
        care of model weight update cycle
        """
        print(f"Training epoch {self.epoch}...")
        self.model.train()

        # Loop over our minibatches
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # We have our data in batch variable. Put the slices as 4D Torch Tensors of
            # shape [BATCH_SIZE, 1, PATCH_SIZE, PATCH_SIZE] into variables data and target.
            # Feed data to the model and feed target to the loss function
            #
            data = batch['image'].to(self.device, dtype=torch.float)
            target = batch['seg'].to(self.device)

            prediction = self.model(data)


            # We are also getting softmax'd version of prediction to output a probability map
            # so that we can see how the model converges to the solution
            prediction_softmax = F.softmax(prediction, dim=1)

            loss = self.loss_function(prediction, target[:, 0, :, :])

            # What does each dimension of variable prediction represent?
            # Each dimension is the probability for each pixel of a imput 2D slice for each class

            loss.backward()
            self.optimizer.step()

            if (i % 10) == 0:
                # Output to console on every 10th batch
                print(f"\nEpoch: {self.epoch} Train loss: {loss}, {100*(i+1)/len(self.train_loader):.1f}% complete")

                counter = 100*self.epoch + 100*(i/len(self.train_loader))

                log_to_tensorboard(
                    self.tensorboard_train_writer,
                    loss,
                    data,
                    target,
                    prediction_softmax,
                    prediction,
                    counter)

            print(".", end='')

        print("\nTraining complete")

    def validate(self):
        """
        This method runs validation cycle, using same metrics as
        Train method. Note that model needs to be switched to eval
        mode and no_grad needs to be called so that gradients do not
        propagate
        """
        print(f"Validating epoch {self.epoch}...")

        # Turn off gradient accumulation by switching model to "eval" mode
        self.model.eval()
        loss_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):

                # Compute loss on a validation sample
                data = batch['image'].to(self.device, dtype=torch.float)
                target = batch['seg'].to(self.device)

                prediction = self.model(data)

                # We are also getting softmax'd version of prediction to output a probability map
                # so that we can see how the model converges to the solution
                prediction_softmax = F.softmax(prediction, dim=1)

                loss = self.loss_function(prediction, target[:, 0, :, :])

                print(f"Batch {i}. Data shape {data.shape} Loss {loss}")

                # We report loss that is accumulated across all of validation set
                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        log_to_tensorboard(
            self.tensorboard_val_writer,
            np.mean(loss_list),
            data,
            target,
            prediction_softmax,
            prediction,
            (self.epoch+1) * 100)
        print(f"Validation complete")

    def save_predictions(self):
        """
        Saves model predicted images in results directory
        """
        print("Save image predictions")

        # Prepare model for inference
        self.model.eval()
        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)
        # Get first test data volume
        first_test_data = self.test_data[0]
        # Get the model predictions
        pred_label = inference_agent.single_volume_inference(first_test_data["image"])
        # Calculate middle slice indice
        axial_middle_index = int(pred_label.shape[0] / 2)
        # Create middle slice images for these volumes for mri image, target and predictions for this epoch
        image = (first_test_data["image"][axial_middle_index] * 255).astype(np.uint8)
        label = (first_test_data["seg"][axial_middle_index] * 255).astype(np.uint8)
        prediction = (pred_label[axial_middle_index] * 255).astype(np.uint8)
        # Convert from numpy array to image objects
        image = Image.fromarray(image)
        label = Image.fromarray(label)
        prediction = Image.fromarray(prediction)
        # Save images
        image.save(self.out_images_dir + '/Epoch' + str(self.epoch) + '-image.png', cmap='Greys')
        label.save(self.out_images_dir + '/Epoch' + str(self.epoch) + '-label.png', cmap='Greys')
        prediction.save(self.out_images_dir + '/Epoch' + str(self.epoch) + '-prediction.png', cmap='Greys')

    def save_model_parameters(self):
        """
        Saves model parameters to a file in results directory
        """
        path = os.path.join(self.out_dir, "model.pth")

        torch.save(self.model.state_dict(), path)

    def load_model_parameters(self, path=''):
        """
        Loads model parameters from a supplied path or a
        results directory
        """
        if not path:
            model_path = os.path.join(self.out_dir, "model.pth")
        else:
            model_path = path

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise Exception(f"Could not find path {model_path}")

    def run_test(self):
        """
        This runs test cycle on the test dataset.
        Note that process and evaluations are quite different
        Here we are computing a lot more metrics and returning
        a dictionary that could later be persisted as JSON
        """
        print("Testing...")
        self.model.eval()

        # In this method we will be computing metrics that are relevant to the task of 3D volume
        # segmentation. Therefore, unlike train and validation methods, we will do inferences
        # on full 3D volumes, much like we will be doing it when we deploy the model in the
        # clinical environment.

        # Instantiate inference agent
        inference_agent = UNetInferenceAgent(model=self.model, device=self.device)

        out_dict = {}
        out_dict["volume_stats"] = []
        dc_list = []
        jc_list = []

        # for every in test set
        for i, x in enumerate(self.test_data):
            pred_label = inference_agent.single_volume_inference(x["image"])

            # We compute and report Dice and Jaccard similarity coefficients which
            # assess how close our volumes are to each other

            dc = Dice3d(pred_label, x["seg"])
            jc = Jaccard3d(pred_label, x["seg"])
            dc_list.append(dc)
            jc_list.append(jc)

            out_dict["volume_stats"].append({
                "filename": x['filename'],
                "dice": dc,
                "jaccard": jc
                })
            print(f"{x['filename']} Dice {dc:.4f} Jaccard {dc:.4f} {100*(i+1)/len(self.test_data):.2f}% complete")

        mean_dice = np.mean(dc_list)
        mean_jaccard = np.mean(jc_list)

        print(f" Mean Dice {mean_dice:.4f} Mean Jaccard {mean_jaccard:.4f}")

        out_dict["overall"] = {
            "mean_dice": mean_dice,
            "mean_jaccard": mean_jaccard}

        print("\nTesting complete.")
        return out_dict

    def run(self):
        """
        Kicks off train cycle and writes model parameter file at the end
        """
        self._time_start = time.time()

        print("Experiment started.")

        # Iterate over epochs
        for self.epoch in range(self.n_epochs):
            self.train()
            self.validate()
            self.save_predictions()

        # save model for inferencing
        self.save_model_parameters()

        self._time_end = time.time()
        print(f"Run complete. Total time: {time.strftime('%H:%M:%S', time.gmtime(self._time_end - self._time_start))}")
