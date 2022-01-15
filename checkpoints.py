import os
import torch
from PIL import Image
import numpy as np
from torchvision.utils import save_image

import config
import model


def save_checkpoint(model, optimizer, filename = "checkpoint.pth.tar"):
    # Printing a message confirming checkpoint being saved
    print("=====> Saving checkpoint.")

    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, filename)
    # Printing a message to confirm that checkpoint has been saved
    print("=====> Checkpoint saved.")


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    # Printing a message confirming checkpoint being loaded
    print("=====> Loading checkpoint.")

    checkpoint = torch.load(checkpoint_file, map_location = config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Updating the learning rate in order to avoid the previous learning rate being used here as well
    for param in optimizer.param_groups:
        param["lr"] = lr

    # Printing a message to confirm that the process has been completed
    print("=====> Checkpoint loaded and learning rate updated.")
