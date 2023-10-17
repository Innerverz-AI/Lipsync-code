import abc
import os
import random

import numpy as np
import torch
from core.dataset import MyDataset, divide_datasets
from lib import utils
from torch.utils.data import DataLoader

# from packages import Ranger
from tqdm import tqdm


class ModelInterface(metaclass=abc.ABCMeta):
    """
    Base class for face GAN models. This base class can also be used
    for neural network models with different purposes if some of concrete methods
    are overrided appropriately. Exceptions will be raised when subclass is being
    instantiated but abstract methods were not implemented.
    """

    def __init__(self, CONFIG, accelerator):
        """
        When overrided, super call is required.
        """

        self.accelerator = accelerator

        self.G = None
        self.D = None
        self.S = None

        self.CONFIG = CONFIG
        self.train_dict = {}
        self.valid_dict = {}
        self.test_dict = {}

        self.SetupModel()

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def SetupModel(self):
        self.set_seed(42)
        self.declare_networks()
        self.set_optimizers()

        if self.CONFIG["CKPT"]["TURN_ON"]:
            self.load_checkpoint()

        divide_datasets(self, self.CONFIG)
        self.set_datasets()
        self.set_loss_collector()

        (
            self.G,
            self.D,
            self.opt_G,
            self.opt_D,
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.accelerator.prepare(
            self.G,
            self.D,
            self.opt_G,
            self.opt_D,
            self.train_dataloader,
            self.valid_dataloader,
        )
        self.G = self.G.module
        self.D = self.D.module

        if self.accelerator.is_main_process:
            print(f"Model {self.CONFIG['BASE']['MODEL_ID']} has successively created")

    def update_net(self, optimizer, loss):
        optimizer.zero_grad()
        self.accelerator.backward(loss)
        optimizer.step()

    def load_next_batch(self, dataloader, iterator, mode):
        """
        Load next batch of source image, target image, and boolean values that denote
        if source and target are identical.
        """
        try:
            batch_data = next(iterator)
            if len(batch_data) == 1:
                batch_data = batch_data[0].cuda()
            else:
                tmp_batch = []
                for data in batch_data:
                    if isinstance(data, dict):
                        for key in data.keys():
                            data[key] = data[key].cuda()
                        tmp_batch.append(data)
                    else:
                        tmp_batch.append(data.cuda())
                batch_data = tmp_batch

        except StopIteration:
            self.__setattr__(mode + "_iterator", iter(dataloader))
            batch_data = next(self.__getattribute__(mode + "_iterator"))
            if len(batch_data) == 1:
                batch_data = batch_data[0].cuda()
            else:
                tmp_batch = []
                for data in batch_data:
                    if isinstance(data, dict):
                        for key in data.keys():
                            data[key] = data[key].cuda()
                        tmp_batch.append(data)
                    else:
                        tmp_batch.append(data.cuda())
                batch_data = tmp_batch

        return batch_data

    def set_datasets(self):
        """
        Initialize dataset using the dataset paths specified in the command line arguments.
        """
        self.train_dataset = MyDataset(self.CONFIG, self.train_dataset_dict)
        self.valid_dataset = MyDataset(self.CONFIG, self.valid_dataset_dict)

        self.set_train_data_iterator()
        self.set_valid_data_iterator()

        if self.accelerator.is_main_process:
            print(
                f"Dataset of {self.train_dataset.__len__()} images constructed for the TRAIN."
            )
            print(
                f"Dataset of {self.valid_dataset.__len__()} images constructed for the VALID."
            )

    def set_train_data_iterator(self):
        """
        Construct sampler according to number of GPUs it is utilizing.
        Using self.dataset and sampler, construct dataloader.
        Store Iterator from dataloader as a member variable.
        """
        # sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.CONFIG['BASE']['USE_MULTI_GPU'] else None
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.CONFIG["BASE"]["BATCH_PER_GPU"],
            pin_memory=True,
            num_workers=8,
            drop_last=True,
        )
        self.train_iterator = iter(self.train_dataloader)

    def set_valid_data_iterator(self):
        """
        Predefine test images only if args.valid_dataset_root is specified.
        These images are anchored for checking the improvement of the model.
        """
        self.valid_dataloader = DataLoader(
            self.valid_dataset, batch_size=1, pin_memory=True, drop_last=True
        )
        self.valid_iterator = iter(self.valid_dataloader)

    @abc.abstractmethod
    def declare_networks(self):
        """
        Construct networks, send it to GPU, and set training mode.
        Networks should be assigned to member variables.

        eg. self.D = Discriminator(input_nc=3).cuda(self.gpu).train()
        """
        pass

    def save_checkpoint(self):
        """
        Save model and optimizer parameters.
        """
        if self.accelerator.is_main_process:
            print(
                f"\nCheckpoints are succesively saved in {self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['BASE']['RUN_ID']}/ckpt/\n"
            )

        ckpt_dict = {}
        ckpt_dict["global_step"] = self.CONFIG["BASE"]["GLOBAL_STEP"]

        ckpt_dict["model"] = self.G.state_dict()
        ckpt_dict["optimizer"] = self.opt_G.state_dict()
        torch.save(
            ckpt_dict,
            f"{self.CONFIG['BASE']['SAVE_ROOT_CKPT']}/G_{str(self.CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}.pt",
        )  # max 99,999,999
        torch.save(ckpt_dict, f"{self.CONFIG['BASE']['SAVE_ROOT_CKPT']}/G_latest.pt")

        if self.D:
            ckpt_dict["model"] = self.D.state_dict()
            ckpt_dict["optimizer"] = self.opt_D.state_dict()
            torch.save(
                ckpt_dict,
                f"{self.CONFIG['BASE']['SAVE_ROOT_CKPT']}/D_{str(self.CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}.pt",
            )  # max 99,999,999
            torch.save(
                ckpt_dict, f"{self.CONFIG['BASE']['SAVE_ROOT_CKPT']}/D_latest.pt"
            )

    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """
        FLAG = False
        for run_id in os.listdir("./train_results"):
            if int(run_id[:3]) == self.CONFIG["CKPT"]["ID_NUM"]:
                self.CONFIG["CKPT"]["ID"] = run_id
                FLAG = True

        assert FLAG, "ID_NUM is wrong"

        ckpt_step = (
            "latest"
            if self.CONFIG["CKPT"]["STEP"] is None
            else str(self.CONFIG["CKPT"]["STEP"]).zfill(8)
        )

        ckpt_path_G = f"{self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['CKPT']['ID']}/ckpt/G_{ckpt_step}.pt"
        ckpt_dict_G = torch.load(ckpt_path_G, map_location=torch.device("cuda"))
        self.G.load_state_dict(ckpt_dict_G["model"], strict=False)
        self.opt_G.load_state_dict(ckpt_dict_G["optimizer"])

        if self.D:
            ckpt_path_D = f"{self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['CKPT']['ID']}/ckpt/D_{ckpt_step}.pt"
            ckpt_dict_D = torch.load(ckpt_path_D, map_location=torch.device("cuda"))
            self.D.load_state_dict(ckpt_dict_D["model"], strict=False)
            self.opt_D.load_state_dict(ckpt_dict_D["optimizer"])

        self.CONFIG["BASE"]["GLOBAL_STEP"] = ckpt_dict_G["global_step"]

        if self.accelerator.is_main_process:
            print(
                f"Pretrained parameters are succesively loaded from {self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['CKPT']['ID']}/ckpt/"
            )

    def set_optimizers(self):
        if self.CONFIG["OPTIMIZER"]["TYPE"] == "Adam":
            self.opt_G = torch.optim.Adam(
                self.G.parameters(),
                lr=self.CONFIG["OPTIMIZER"]["LR_G"],
                betas=self.CONFIG["OPTIMIZER"]["BETA"],
            )
            if self.D:
                self.opt_D = torch.optim.Adam(
                    self.D.parameters(),
                    lr=self.CONFIG["OPTIMIZER"]["LR_D"],
                    betas=self.CONFIG["OPTIMIZER"]["BETA"],
                )

    @abc.abstractmethod
    def set_loss_collector(self):
        """
        Set self.loss_collector as an implementation of lib.loss.LossInterface.
        """
        pass

    @property
    @abc.abstractmethod
    def loss_collector(self):
        """
        loss_collector should be an implementation of lib.loss.LossInterface.
        This property should be assigned in self.set_loss_collector.
        """
        pass

    @abc.abstractmethod
    def go_step(self):
        """
        Implement a single iteration of training. This will be called repeatedly in a loop.
        This method should return list of images that was created during training.
        Returned images are passed to self.save_image and self.save_image is called in the
        training loop preiodically.
        """
        pass

    @abc.abstractmethod
    def do_validation(self):
        """
        Test the model using a predefined valid set.
        This method includes util.save_image and returns nothing.
        """
        pass

    def do_validation(self, val_step):
        self.valid_images = []
        self.set_networks_eval_mode()

        (
            self.loss_collector.loss_dict["valid_L_G"],
            self.loss_collector.loss_dict["valid_L_D"],
        ) = (0.0, 0.0)
        # pbar = tqdm(range(len(self.valid_dataloader)), desc='Run validate..')
        pbar = tqdm(range(val_step), desc="Run validate..")
        for _ in pbar:
            batch_data_bundle = self.load_next_batch(
                self.valid_dataloader, self.valid_iterator, "valid"
            )

            for data_name, batch_data in zip(self.batch_data_names, batch_data_bundle):
                self.valid_dict[data_name] = batch_data

            with torch.no_grad():
                self.run_G(self.valid_dict)
                self.run_D(self.valid_dict)
                self.loss_collector.get_loss_G(
                    self.valid_dict, model=self.G, prev_model=self.G_pre, valid=True
                )
                self.loss_collector.get_loss_D(self.valid_dict, valid=True)

            if len(self.valid_images) < 8:
                utils.stack_image_grid(
                    [
                        self.valid_dict[data_name]
                        for data_name in self.saving_data_names
                    ],
                    self.valid_images,
                )

        self.loss_collector.loss_dict["valid_L_G"] /= len(self.valid_dataloader)
        self.loss_collector.loss_dict["valid_L_D"] /= len(self.valid_dataloader)
        self.loss_collector.val_print_loss()

        self.valid_images = torch.cat(self.valid_images, dim=-1)

        self.set_networks_train_mode()
