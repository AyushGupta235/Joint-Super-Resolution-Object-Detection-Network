import torch
from torch.utils.data import DataLoader

from model import Discriminator1, Discriminator2, Generator, YOLOv5,  mseLoss, Perceptual_Loss, Adversarial_Loss, Detection_Loss, TotalLoss
from dataset import ImageFolder
import config
import checkpoints


def test(loader, gen, disc2, detector, ):


def run_test():
    dataset = ImageFolder(root_dir = config.ORG_DIR)
    """Check pin_memory and num_workers"""
    loader = DataLoader(
        dataset,
        batch_size = config.TEST_BATCH_SIZE,
        shuffle = False,
        pin_memory = True, 
        num_workers = config.NUM_WORKERS,
    )

    gen = Generator().to(config.DEVICE)
    gen.load_state_dict(torch.load(config.MODEL_GEN))

    disc2 = Discriminator2().to(config.DEVICE)
    disc2.load_state_dict(torch.load(config.MODEL_DISC2))

    det = YOLOv5().to(config.DEVICE)
    det.load_state_dict(torch.load(config.MODEL_DET))