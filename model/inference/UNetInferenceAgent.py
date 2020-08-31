"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """

        # Set model to eval no gradients
        self.model.eval()
        # Initialise slices with zeros
        slices = np.zeros(volume.shape)
        # Reshape volume to conform to required model patch size
        volume = med_reshape(volume, new_shape=(volume.shape[0], self.patch_size, self.patch_size))

        # For each x slice in the volume
        for x_index in range(volume.shape[0]):
            # Get the x slice
            x_slice = volume[x_index,:,:].astype(np.single)
            # Convert to tensor
            tensor_x_slice = torch.from_numpy(x_slice).unsqueeze(0).unsqueeze(0).to(self.device)
            # Pass slice through model to get predictions
            predictions = self.model(tensor_x_slice)
            # Resize predictions
            pred_resized = np.squeeze(predictions.cpu().detach())
            # Append predictions
            slices[x_index,:,:] = torch.argmax(pred_resized, dim=0)

        # Return volume of predictions
        return slices

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # Create mask for each slice across the X (0th) dimension. After
        # that, put all slices into a 3D Numpy array. We can verify if the method is
        # correct by running it on one of the volumes in your training set and comparing
        # with the label in 3D Slicer.

        # Set model to eval no gradients
        self.model.eval()
        # Initialise slices with zeros
        slices = np.zeros(volume.shape)

        # For each x slice in the volume
        for x_index in range(volume.shape[0]):
            # Get the x slice
            x_slice = volume[x_index,:,:].astype(np.single)
            # Convert to tensor
            tensor_x_slice = torch.from_numpy(x_slice).unsqueeze(0).unsqueeze(0).to(self.device)
            # Pass slice through model to get predictions
            predictions = self.model(tensor_x_slice)
            # Resize predictions
            pred_resized = np.squeeze(predictions.cpu().detach())
            # Append predictions
            slices[x_index,:,:] = torch.argmax(pred_resized, dim=0)

        # Return volume of predictions
        return slices
