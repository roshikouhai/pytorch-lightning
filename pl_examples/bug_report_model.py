import gc
import os

import torch
from torch.utils.data import Dataset

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def loss(self, batch, prediction):
        # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def training_step_end(self, training_step_outputs):
        return training_step_outputs

    def training_epoch_end(self, outputs) -> None:
        self.trainer.accelerator.barrier()
        print(f"rank {self.trainer.global_rank} exp version: {os.environ.get('PL_EXP_VERSION')}", self.logger.version)
        torch.stack([x["loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x['x'] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        self.log('fake_test_acc', loss)
        return {"y": loss}

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["y"] for x in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


def run():
    train_data = torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=2, num_workers=0)
    val_data = torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=2, num_workers=0)
    test_data = torch.utils.data.DataLoader(RandomDataset(32, 64), batch_size=2, num_workers=0)

    for i in range(3):

        if i > 0:
            logger = TensorBoardLogger(save_dir=".", name="lightning_logs")
        else:
            logger = True

        model = BoringModel()
        trainer = Trainer(
            max_epochs=1, progress_bar_refresh_rate=0, accelerator="ddp", gpus=2, weights_summary=None, logger=logger
        )
        print(
            f"iteration {i}, before fit, rank {trainer.global_rank}, "
            f"logger: {trainer.logger.version}, "
            f"env: {os.environ.get('PL_EXP_VERSION')}"
            f"logdir: {trainer.log_dir}"
            f"save_dir: {trainer.logger.save_dir}"
        )

        trainer.fit(model, train_data, val_data)
        print(
            f"iteration {i}, after fit, rank {trainer.global_rank}, "
            f"logger: {trainer.logger.version}, "
            f"env: {os.environ.get('PL_EXP_VERSION')}"
            f"logdir: {trainer.log_dir}"
            f"save_dir: {trainer.logger.save_dir}"
        )

        trainer.test(test_dataloaders=test_data)

        trainer.accelerator.barrier()

        print(
            f"iteration {i}, after test, rank {trainer.global_rank}, "
            f"logger: {trainer.logger.version}, "
            f"env: {os.environ.get('PL_EXP_VERSION')}"
            f"logdir: {trainer.log_dir}"
            f"save_dir: {trainer.logger.save_dir}"
        )
        del trainer, model
        gc.collect()


if __name__ == '__main__':
    run()
