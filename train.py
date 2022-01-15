from PIL.Image import Image
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Discriminator1, Discriminator2, Generator, YOLOv5,  mseLoss, Perceptual_Loss, Adversarial_Loss, Detection_Loss, TotalLoss
from dataset import ImageFolder
import config
import checkpoints

def train_disjoint(loader, gen, disc2, detector, true_label, optimizer_gen, optimizer_disc2, optimizer_det, mse, bce, vgg):
    loop = tqdm(loader, leave = True)

    for _, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        # Train Discriminators
        fake_img = gen(low_res)
        disc2_real = disc2(high_res)
        disc2_fake = disc2(fake_img.detach())

        disc2_loss_real = bce(
            disc2_real, torch.ones_like((disc2_real) - 0.1 * torch.rand_like(disc2_real))
        )
        disc2_loss_fake = bce(
            disc2_fake, torch.zeros_like(disc2_fake)
        )
        loss_disc2 = disc2_loss_fake + disc2_loss_real

        optimizer_disc2.zero_grad()
        loss_disc2.backward()
        optimizer_disc2.step()

        # Training Generator
        disc2_fake = disc2(fake_img)
        
        l2_loss = mse(fake_img, high_res)
        adv_loss = Adversarial_Loss(fake_img, high_res)
        vgg_loss = vgg(fake_img, high_res)

        gen_loss = l2_loss + adv_loss + vgg_loss

        optimizer_gen.zero_grad()
        gen_loss.backward()
        optimizer_gen.step()

        # Training Object Detector
        det = detector(fake_img)

        det_loss = Detection_Loss(out = det, label = true_label, obj = True)

        optimizer_det.zero_grad()
        det_loss.backward()
        optimizer_det.step()

def train_joint(loader, gen, disc2, detector, true_label, optimizer, mse, bce, vgg):
    loop = tqdm(loader, leave = True)

    for _, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        # Train Discriminators
        fake_img = gen(low_res)
        disc2_real = disc2(high_res)
        disc2_fake = disc2(fake_img.detach())

        disc2_loss_real = bce(
            disc2_real, torch.ones_like((disc2_real) - 0.1 * torch.rand_like(disc2_real))
        )
        disc2_loss_fake = bce(
            disc2_fake, torch.zeros_like(disc2_fake)
        )
        loss_disc2 = disc2_loss_fake + disc2_loss_real

        # Training Generator
        disc2_fake = disc2(fake_img)
        
        l2_loss = mse(fake_img, high_res)
        adv_loss = Adversarial_Loss(fake_img, high_res)
        vgg_loss = vgg(fake_img, high_res)

        # Training Object Detector
        det = detector(fake_img)

        det_loss = Detection_Loss(out = det, label = true_label, obj = True)

        # Compiling all the losses
        total_loss = l2_loss + (config.BETA * adv_loss) + (config.ALPHA * vgg_loss) + (config.GAMMA * det_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def run_disjoint():
    dataset = ImageFolder(root_dir = "")
    loader = DataLoader(
        dataset,
        batch_size = config.SR_BATCH_SIZE,
        shuffle = True,
        pin_memory = True,
        num_workers = config.NUM_WORKERS,
    )

    gen = Generator().to(config.DEVICE)
    #disc1 = Discriminator1().to(config.DEVICE)
    disc2 = Discriminator2().to(config.DEVICE)
    det = YOLOv5().to(config.DEVICE)

    optimizer_gen = optim.Adam(gen.parameters(), lr = config.SR_LEARNING_RATE, betas = (0.9, 0.999))
    #optimizer_disc1 = optim.Adam(disc1.parameters(), lr = config.SR_LEARNING_RATE, betas = (0.9, 0.999))
    optimizer_disc2 = optim.Adam(disc2.parameters(), lr = config.SR_LEARNING_RATE, betas = (0.9, 0.999))
    optimizer_det = optim.Adam(det.parameters(), lr = config.DET_LEARNING_RATE_1, betas = (0.9, 0.999))
    
    mse = mseLoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = Perceptual_Loss()

    if config.LOAD_MODEL:
        checkpoints.load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            optimizer_gen,
            config.SR_LEARNING_RATE,
        )

        #load_checkpoint(
        #    config.CHECKPOINT_DISC1,
        #    disc1,
        #    optimizer_disc1,
        #    config.SR_LEARNING_RATE,
        #)

        checkpoints.load_checkpoint(
            config.CHECKPOINT_DISC2,
            disc2,
            optimizer_disc2,
            config.SR_LEARNING_RATE,
        )

        checkpoints.load_checkpoint(
            config.CHECKPOINT_DET,
            det,
            optimizer_det,
            config.SR_LEARNING_RATE,
        )

    for _ in range(config.EPOCHS):
        train_disjoint(
            loader = loader,
            gen = gen,
            disc2 = disc2,
            det = det,
            optimizer_gen = optimizer_gen,
            optimizer_disc2 = optimizer_disc2,
            optimizer_det = optimizer_det,
            mse = mse,
            bce = bce,
            vgg = vgg_loss,
        )

        if config.SAVE_MODEL:
            checkpoints.save_checkpoint(gen, optimizer_gen, filename = config.CHECKPOINT_GEN)
            checkpoints.save_checkpoint(disc2, optimizer_disc2, filename = config.CHECKPOINT_DISC2)

    torch.save(gen.state_dict(), config.MODEL_GEN)
    torch.save(disc2.state_dict(), config.MODEL_DISC2)
    torch.save(det.state_dict(), config.MODEL_DET)


def run_joint():
    dataset = ImageFolder(root_dir = "")
    loader = DataLoader(
        dataset,
        batch_size = config.SR_BATCH_SIZE,
        shuffle = True,
        pin_memory = True,
        num_workers = config.NUM_WORKERS,
    )

    gen = Generator().to(config.DEVICE)
    gen.load_state_dict(torch.load(config.MODEL_GEN))
    #disc1 = Discriminator1().to(config.DEVICE)
    disc2 = Discriminator2().to(config.DEVICE)
    disc2.load_state_dict(torch.load(config.MODEL_DISC2))
    det = YOLOv5().to(config.DEVICE)
    det.load_state_dict(torch.load(config.MODEL_DET))

    optimizer_gen = optim.Adam(gen.parameters(), lr = config.SR_LEARNING_RATE, betas = (0.9, 0.999))
    #optimizer_disc1 = optim.Adam(disc1.parameters(), lr = config.SR_LEARNING_RATE, betas = (0.9, 0.999))
    optimizer_disc2 = optim.Adam(disc2.parameters(), lr = config.SR_LEARNING_RATE, betas = (0.9, 0.999))
    optimizer_det = optim.Adam(det.parameters(), lr = config.DET_LEARNING_RATE_1, betas = (0.9, 0.999))
    
    optimizer = optim.Adam(list(gen.parameters()) + list(disc2.parameters()) + list(det.parameters()), lr = config.JT_LEARNING_RATE, betas = (0.9, 0.999))
    mse = mseLoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = Perceptual_Loss()

    if config.LOAD_MODEL:
        checkpoints.load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            optimizer_gen,
            config.JT_LEARNING_RATE,
        )

        #load_checkpoint(
        #    config.CHECKPOINT_DISC1,
        #    disc1,
        #    optimizer_disc1,
        #    config.SR_LEARNING_RATE,
        #)

        checkpoints.load_checkpoint(
            config.CHECKPOINT_DISC2,
            disc2,
            optimizer_disc2,
            config.JT_LEARNING_RATE,
        )

        checkpoints.load_checkpoint(
            config.CHECKPOINT_DET,
            det,
            optimizer_det,
            config.JT_LEARNING_RATE,
        )

    for _ in range(config.EPOCHS):
        train_joint(
            loader = loader,
            gen = gen,
            disc2 = disc2,
            det = det,
            optimizer = optimizer,
            mse = mse,
            bce = bce,
            vgg = vgg_loss,
        )

        if config.SAVE_MODEL:
            checkpoints.save_checkpoint(gen, optimizer_gen, filename = config.CHECKPOINT_GEN)
            checkpoints.save_checkpoint(disc2, optimizer_disc2, filename = config.CHECKPOINT_DISC2)


if __name__ == "__main__":
    if config.MODE == "DISJOINT":
        run_disjoint()
    elif config.MODE == "JOINT":
        run_joint()