import pytorch_lightning as pl
import numpy as np
import random
import os
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import json

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import datasets, transforms

from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy

import torch
import torch.nn as nn

from pt_tnn.temporal_graph import TemporalGraph
from pt_tnn.recurrent_module import RecurrentModuleRGC


class MakeMovie(nn.Module):
    """
    A custom transform that repeats the given tensor for `image_off` frames,
    then inserts blank frames (filled with 0.5) for the remaining `times - image_off` frames.

    Args:
        times (int): Total number of frames.
        image_off (int): Number of frames to show the original image/tensor before switching to blank frames.
    """

    def __init__(self, times: int, image_off: int):
        super(MakeMovie, self).__init__()
        self.times = times
        self.image_off = image_off

    def forward(self, ims: torch.Tensor):
        """
        Args:
            ims (torch.Tensor): The input tensor (e.g., a single image or batch of images).

        Returns:
            torch.Tensor: bs, t=self.image_off+(self.times - self.image_off), ...
        """
        shape = ims.shape
        if len(shape) == 4:
            blank = torch.full_like(ims, 0.5)
            pres = [ims] * self.image_off + [blank] * (self.times - self.image_off)
            return torch.stack(pres, dim=1)
        elif len(shape) == 5:
            assert (
                    shape[1] == self.times
            ), f"T={self.times} but got input time dimension={shape[1]}"
            ims[:, self.image_off:, ...] = 0.5
            return ims
        else:
            print(
                f"currently only (bs, C, H, W) or (bs, T, C, H, W) is supported, but the input shape={shape}"
            )
            raise Exception

    def __repr__(self):
        return (
                self.__class__.__name__ + f"(times={self.times}-image-off={self.image_off})"
        )


class MetricRecorder:
    def __init__(self, name, verbose):
        self.name = name
        self.verbose = verbose

        # metrics to record
        self.total_loss = 0.0
        self.total_acc = 0.0
        self.total_samples = 0

    def reset(self):
        # metrics to record
        self.total_loss = 0.0
        self.total_samples = 0
        self.total_acc = 0

    def update(self, loss, num_samples, acc=None):
        if loss is not None:
            self.total_loss += loss.cpu().detach().item() * num_samples
        self.total_samples += num_samples
        if acc is not None:
            self.total_acc += acc.cpu().detach().item() * num_samples

    def fetch_and_print(self, epoch=None, lr=None):
        avr_loss = self.total_loss / self.total_samples
        avr_acc = self.total_acc / self.total_samples

        if self.verbose:
            print()
            print(
                f"{self.name} | mean loss {avr_loss:5.2f} | mean acc {avr_acc:5.2f} | lr {lr} | epoch {epoch}"
            )
        return {
            "avr_loss": avr_loss,
            "avr_acc": avr_acc,
        }


class TNNModel(LightningModule):
    def __init__(
            self,
            model,
            n_times,
            num_classes,
            input_shape,
            lr=0.1,
            step_size=30,
            epochs=100,
            warmup_epochs=10,
            weight_decay=0.0001,
            momentum=0.9,
    ):
        super(TNNModel, self).__init__()

        self.model = model
        self.n_times = n_times
        self.input_shape = input_shape

        # train configurations
        self.lr = lr
        self.step_size = step_size
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum

        # metrics to compute (e.g., loss, acc)
        self.loss = torch.nn.CrossEntropyLoss()
        self.top1_acc = MulticlassAccuracy(
            num_classes=num_classes, average="micro", top_k=1
        )
        self.top5_acc = MulticlassAccuracy(
            num_classes=num_classes, average="micro", top_k=5
        )

        # train & validation dynamics
        self.train_recorder = MetricRecorder(name="train_dynamics", verbose=True)
        self.val_recorder = MetricRecorder(name="val_dynamics", verbose=True)
        self.train_losses = []
        self.val_losses = []

    def configure_optimizers(self):
        all_parameters = list(self.model.parameters())

        optimizer = torch.optim.SGD(
            all_parameters,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.step_size,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor for schedulers like `ReduceLROnPlateau`
                "interval": "epoch",  # The unit of the scheduler's step size, could also be 'step'. 'epoch' updates
                # the scheduler on epoch end whereas 'step' updates it after a optimizer update.
                "frequency": 1,  # How many epochs/steps should pass between calls to `scheduler.step()`. 1 corresponds
                # to updating the learning rate after every epoch/step.
            },
        }

    # Dummy forward pass to initialize uninitialized parameters (e.g., LazyLayers) before DDP
    def setup(self, stage=None):
        dummy_input = torch.randn(
            (5, self.n_times) + tuple(self.input_shape), device=self.device
        )  # batch_size=5

        self.model.to(self.device)

        print("=" * 20, "dummy input shape", dummy_input.shape)
        self.forward(dummy_input)

    def forward(self, x):
        # x: batch of image(-alike) inputs (N, T, C, H, W) or (N, C, H, W). (batch, time, channel, height, width)
        return (
            self.model(x, n_times=self.n_times).squeeze(-1).squeeze(-1)
        )  # (bs, num_class)

    def compute_loss(self, pred, targets):
        return self.loss(pred, targets)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        pred = self.forward(inputs)
        loss = self.compute_loss(pred=pred, targets=targets)
        self.train_recorder.update(loss=loss, num_samples=inputs.shape[0])

        if not torch.isfinite(loss).item():
            # 1) mark the run as failed in the logs (optional)
            self.log(
                "train_loss_nonfinite",
                loss,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
            )
            # 2) ask the Trainer to shut down gracefully
            self.trainer.should_stop = True

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        # (bs, 22, 30, 5, 7)
        pred = self.forward(inputs)  # (bs, N, C, H, W) ---> (bs, num_class)

        val_loss = self.compute_loss(pred=pred, targets=targets)
        val_acc = self.top1_acc(preds=pred, target=targets)

        self.val_recorder.update(
            loss=val_loss, num_samples=inputs.shape[0], acc=val_acc
        )

        return val_loss

    def test_step(self, batch, batch_idx):
        # Defines a single test step. Similar to validation_step, but for test data.
        inputs, targets = batch
        # Forward pass to get predictions
        pred = self.forward(inputs)
        # Compute accuracy
        test_acc_top1 = self.top1_acc(preds=pred, target=targets)
        test_acc_top5 = self.top5_acc(preds=pred, target=targets)
        # Log the test accuracy
        self.log("test_acc_top1", test_acc_top1, sync_dist=True, prog_bar=True)
        self.log("test_acc_top5", test_acc_top5, sync_dist=True, prog_bar=True)

        return {"test_acc_top1": test_acc_top1, "test_acc_top5": test_acc_top5}

    def on_train_epoch_start(self) -> None:
        self.train_recorder.reset()  # reset the recorded dynamics for the next epoch

    def on_train_epoch_end(self) -> None:
        lr = self.lr_schedulers().get_last_lr()[0]
        train_metric = self.train_recorder.fetch_and_print(
            epoch=self.current_epoch, lr=lr
        )
        self.log("train_loss", train_metric["avr_loss"], sync_dist=True)
        self.train_losses.append(train_metric["avr_loss"])

    def on_validation_epoch_start(self) -> None:
        self.val_recorder.reset()  # reset the recorded dynamics for the next epoch

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            val_metric = self.val_recorder.fetch_and_print(
                epoch=self.current_epoch, lr=None
            )
            self.log("val_loss", val_metric["avr_loss"], sync_dist=True)
            self.log("val_acc", val_metric["avr_acc"], sync_dist=True)
            self.val_losses.append(val_metric["avr_loss"])

    def on_save_checkpoint(self, checkpoint):
        # Save the lists of train and val losses
        checkpoint["train_losses"] = self.train_losses
        checkpoint["val_losses"] = self.val_losses

    def on_load_checkpoint(self, checkpoint):
        # Load the lists of train and val losses
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        print("-" * 20)
        print(
            f"getting the train losses of length {len(self.train_losses)} & the val losses of length "
            f"{len(self.val_losses)} from the latest ckpt"
        )
        train_losses_len = len(self.train_losses)
        val_losses_len = len(self.val_losses)
        if (
                train_losses_len > val_losses_len
        ):  # training collapsed after train epoch & before val epoch
            self.train_losses = self.train_losses[:val_losses_len]
        elif val_losses_len > train_losses_len:
            raise Exception  # then sth. is really off
        print("-" * 20)


class ImageNetDataModule(LightningDataModule):
    def __init__(self, batch_size=256, num_workers=32):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dir = (
            "/data/group_data/neuroagents_lab/training_datasets/imagenet_raw/train"
        )
        self.val_dir = (
            "/data/group_data/neuroagents_lab/training_datasets/imagenet_raw/val"
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        # Download the dataset (if needed)
        pass

    def setup(self, stage=None):
        # Split the dataset into train/val datasets
        if stage == "fit" or stage is None:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            self.train_dataset = datasets.ImageFolder(
                root=self.train_dir,
                transform=transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )

            self.val_dataset = datasets.ImageFolder(
                root=self.val_dir,
                transform=transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
            # from https://github.com/pytorch/examples/blob/main/imagenet/main.py

        if stage == "test" or stage is None:
            self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=4 * self.batch_size,
            num_workers=1,  # self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=4 * self.batch_size,
            # num_workers=self.num_workers,
            # pin_memory=True,
        )


def set_seed(seed: int = 42) -> None:
    # https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def main(
        run_name,
        model_config_file,
        num_timesteps,
        lr,
        input_shape,
        batch_size,
        num_devices,
        transform,
):
    set_seed(seed=0)

    assert run_name is not None, "please specify a name for this run"

    model_save_path = f"ckpt/{run_name}"
    results_save_path = f"results/{run_name}"

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    # Initialize the WandbLogger
    os.environ["WANDB_DIR"] = f"{results_save_path}/wandb"
    logger = WandbLogger(
        project="imagenet_tnn", name=run_name, save_dir=results_save_path
    )

    # Create the model
    TG = TemporalGraph(
        model_config_file=model_config_file,
        recurrent_module=RecurrentModuleRGC if 'rgc' in model_config_file else None,
        # (default: None, which means using the RecurrentModule [compatible for models other than RGC] from pt_tnn)
        input_shape=input_shape,
        num_timesteps=num_timesteps,
        transform=transform,
    )

    tnn_model = TNNModel(
        model=TG,
        n_times=num_timesteps,
        lr=lr,
        num_classes=1000,
        input_shape=input_shape,
    )
    data_module = ImageNetDataModule(batch_size=batch_size, num_workers=32)

    # Callback to save the last training checkpoint
    last_ckpt_callback = ModelCheckpoint(
        dirpath=model_save_path,  # Directory to save the last checkpoint
        filename="train_last",  # Name of the last training checkpoint
        save_top_k=0,  # Don't monitor a metric, only save the last
        save_last=True,  # Save the last training checkpoint
        verbose=True,  # Print info about saving
    )

    # Define the checkpoint callback
    val_ckpt_callback = ModelCheckpoint(
        monitor="val_acc",  # The metric to monitor
        mode="max",  # 'min' because you want to save the model with the smallest validation loss
        save_top_k=1,  # Save only the best model
        dirpath=model_save_path,
        filename="val_best",
        # Name of the saved model file
        verbose=True,  # Print information about saving
    )

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[last_ckpt_callback, val_ckpt_callback],
        logger=logger,
        max_epochs=100,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        devices=num_devices,  # or 0 if you're using CPU
        gradient_clip_val=1.0,  # added here to align with shallow rgc training
    )

    trainer.fit(
        model=tnn_model,
        datamodule=data_module,
        # ckpt_path=f"{model_save_path}/last.ckpt",  # uncomment if resume from training
    )

    # using only 1 device for testing following: "It is recommended to test with Trainer(devices=1) since distributed
    # strategies such as DDP use DistributedSampler internally, which replicates some samples to make sure all
    # devices have same batch size in case of uneven inputs. This is helpful to make sure benchmarking for research
    # papers is done the right way." https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html
    test_trainer = pl.Trainer(
        devices=1,
        logger=logger,
        enable_progress_bar=True,
    )

    test_dict = test_trainer.test(
        model=tnn_model,
        datamodule=data_module,
        ckpt_path=f"{model_save_path}/val_best.ckpt",
    )
    print(test_dict)

    # maybe save the test results here
    with open(f"{results_save_path}/test_result.json", "w") as f:
        json.dump(test_dict, f, indent=4)


if __name__ == "__main__":

    run_name = "train_rgc_on_imagenet"  # a special name identifier for wandb, checkpoint and result directories
    model_config_file = "configs/test_rgc_shallow.json"  # model configurations, see more examples under `configs`

    input_shape = [3, 224, 224]  # specify the input shape (C, H, W)
    num_timesteps = 16  # number of unroll times
    transform = MakeMovie(
        times=num_timesteps, image_off=12
    )  # default is None (which means no specific data transform)

    main(
        run_name=run_name,
        model_config_file=model_config_file,
        num_timesteps=num_timesteps,
        lr=0.01,
        input_shape=input_shape,
        batch_size=256,
        num_devices=1,
        transform=transform,
    )
