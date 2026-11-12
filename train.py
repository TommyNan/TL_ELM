import os.path
import time

from sympy.codegen.ast import Raise

from lib.util import set_random_seed
import torch
from datasets.data_loader import get_dataloaders
from Models.model import NeRF2, NeRF2_ESN_V1
from Models.render import get_renderer
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from utils.data_painter import paint_spectrum_compare
# from datasets.dataloader_ss import get_dataloaders
# from Models.ngrf_ss import nGRF
# from train_delete import evaluate
# from utils import get_expon_lr_func
# import torch.nn as nn
import numpy as np
# from utils.loss import calculate_snr

class Runner:
    def __init__(self, **configs):
        super(Runner, self).__init__()
        self.configs = configs
        self.sys_params = configs['sys_params']
        self.data_params = configs['data_params']
        self.model_params = configs['model_params']
        self.train_params = configs['train_params']
        self.logger = configs['logger']
        self.log_dir = configs['log_dir']
        self.load_checkpoint = self.train_params['load_checkpoint']
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.sys_params['device'] == 'cuda' else "cpu")

        # ============================================================
        # Step 1: Initialize Datasets
        # ============================================================
        self.logger.info("STEP 1: Loading Datasets")
        self.train_loader, self.test_loader = get_dataloaders(self.logger,  self.configs)

        # ============================================================
        # Step 2: Initialize Models
        # ============================================================
        self.logger.info("STEP 2: Initialize Models")

        # self.model = NeRF2(self.model_params).to(device=self.device)
        self.model = NeRF2_ESN_V1(self.model_params).to(device=self.device)
        self.logger.info("Successfully Initialized Models, The model architecture is:")
        self.logger.info(self.model.summary())

        # ============================================================
        # Step 3: Initialize Rendering
        # ============================================================
        self.logger.info("STEP 3: Initialize Rendering")
        self.renderer = get_renderer(self.logger, self.configs)

        # ============================================================
        # Step 4: Initialize Optimizer and Scheduler
        # ============================================================
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.train_params['lr']), eps=float(self.train_params['eps']),
                                          weight_decay=float(self.train_params['weight_decay']), betas=(0.9, 0.999),)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                    T_max=int(self.train_params['T_max']),
                                                                    eta_min=float(self.train_params['eta_min']),
                                                                    last_epoch=-1)
        # gamma = (3e-5 / 3e-4) ** (1.0 / self.train_params['total_iters'])
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer,
        #     gamma=gamma
        # )


        # ============================================================
        # Step 4: Initialize Checkpoints
        # ============================================================
        self.checkpoints_dir = os.path.join(self.configs["log_dir"], "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.current_iter = 1
        if self.train_params['load_checkpoint'] or configs['mode'] == 'test':
            self.load_checkpoints()


    def train(self):
        """
        Input: exp_idx
        * exp_1: Comparison among all baselines (insite or gudmundson dataset)
        :return:
        """

        exp_idx = self.configs["exp_name"]
        self.logger.info(f"[Training] Running the Function of Experiment: {self.configs['exp_name']}")
        self.train_exp()


        # exp_name = f"train_exp_{exp_idx}"



        # if hasattr(self, exp_name):
        #     self.logger.info(f"[Training] Running the Function of Experiment: {self.configs['exp_name']}")
        #     # method = getattr(self, exp_name)
        #     method()
        # else:
        #     # raise NotImplementedError(f"Runner has no method named '{exp_name}'")
        #     raise NotImplementedError(f"Runner has no method named '{exp_idx}'")

    def test(self):
        """
        Input: exp_idx
        * exp_1: Comparison among all baselines (insite or gudmundson dataset)
        :return:
        """
        exp_idx = self.configs.get('exp_idx', 1)
        exp_name = f"test_exp_{exp_idx}"

        if hasattr(self, exp_name):
            self.logger.info(f"Running the Function of SEQUA Experiment with trained models: {exp_name}")
            method = getattr(self, exp_name)
            method()
        else:
            raise NotImplementedError(f"Runner has no method named '{exp_name}'")

    def load_checkpoints(self):
        if self.load_checkpoint:
            checkpoint_list = [os.path.join(self.checkpoints_dir, f) for f in sorted(os.listdir(self.checkpoints_dir)) if
                               'tar' in f]

        if len(checkpoint_list) > 0:
            checkpoint_path = checkpoint_list[-1]
            self.logger.info(f'Found checkpoints: {checkpoint_list} and Loading ckpt {checkpoint_path}')
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt['nerf2_network_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=20,eta_min=1e-5)
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.current_iter = ckpt['current_iter']
        else:
            raise NotImplementedError(f"No checkpoint found at {checkpoint_list}")


    def train_exp(self):
        set_random_seed(self.sys_params.get('random_seed', 42))
        if self.train_params['load_checkpoint']:
            self.logger.info(f"Retraining from existing model | Start from Iteration: {self.current_iter}")
        else:
            self.logger.info(f"Training from scratch | Start from Iteration: {self.current_iter}")

        total_iters = self.train_params['total_iters']
        loss_list = []
        while self.current_iter < total_iters:
            with tqdm(total=len(self.train_loader), desc=f"Iteration {self.current_iter}/{total_iters}") as pbar:
                for train_input, train_label in self.train_loader:
                    if self.current_iter > total_iters:
                        break
                    train_input, train_label = train_input.to(self.device), train_label.to(self.device)
                    if self.data_params["type"] == "rfid":
                        rays_o, rays_d, tx_pos = train_input[:, :3], train_input[:, 3:6], train_input[:, 6:9]
                        predict_spectrum = self.renderer.render(self.model, tx_pos, rays_o, rays_d)
                        loss = torch.mean((predict_spectrum - train_label.view(-1)) ** 2)
                    elif self.data_params["type"] == "ble":
                        tx_pos, rays_o, rays_d = train_input[:, :3], train_input[:, 3:6], train_input[:, 6:]
                        predict_rssi = self.renderer.render(self.model, tx_pos, rays_o, rays_d)
                        loss = torch.mean((predict_rssi - train_label.view(-1)) ** 2)
                    elif self.data_params["type"] == "mimo":
                        uplink, rays_o, rays_d = train_input[:, :52], train_input[:, 52:55], train_input[:, 55:]
                        predict_downlink = self.renderer.render(self.model, uplink, rays_o, rays_d)
                        predict_downlink = torch.concat((predict_downlink.real, predict_downlink.imag), dim=-1)
                        loss = torch.mean((predict_downlink - train_label) ** 2)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.current_iter += 1
                    loss_list.append(loss.item())

                    pbar.update(1)
                    pbar.set_description(f"Iteration {self.current_iter}/{total_iters}")
                    pbar.set_postfix_str(f"loss = {loss.item():.6f}, lr = {self.optimizer.param_groups[0]['lr']:.6f}\n")
                    if self.current_iter == 2:
                        total_params = sum(p.numel() for p in self.model.parameters())
                        self.logger.info(f"Total number of parameters: {total_params}")

                    if self.current_iter % self.train_params["save_freq"] == 0:
                        checkpoint_name = self.save_checkpoint()
                        pbar.write(f'Saved checkpoints at {checkpoint_name}')
                        self.logger.info(f"Iter {self.current_iter} loss = {loss.item():.6f}, lr = {self.optimizer.param_groups[0]['lr']:.6f}")

        plt.figure(figsize=(6, 4))
        plt.plot(loss_list)
        plt.xlabel("Iteration")
        plt.ylabel("Training Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Evaluate the model
        self.logger.info("Start evaluation")
        self.model.eval()
        os.makedirs(os.path.join(self.log_dir, 'pred_spectrum'), exist_ok=True)
        pred2next, gt2next = torch.zeros((0)), torch.zeros((0))
        save_img_idx = 0
        all_ssim = []
        all_mae = []
        with torch.no_grad():
            for test_input, test_label in self.test_loader:
                test_input, test_label = test_input.to(self.device), test_label.to(self.device)
                rays_o, rays_d, tx_pos = test_input[:, :3], test_input[:, 3:6], test_input[:, 6:9]
                pred_spectrum = self.renderer.render(self.model, tx_pos, rays_o, rays_d)

                ## save predicted spectrum
                pred_spectrum = pred_spectrum.detach().cpu()
                gt_spectrum = test_label.detach().cpu()
                pred_spectrum = torch.concatenate((pred2next, pred_spectrum), dim=0)
                gt_spectrum = torch.concatenate((gt2next, gt_spectrum), dim=0)
                num_spectrum = len(pred_spectrum) // (360 * 90)
                pred2next = pred_spectrum[num_spectrum*360*90:]
                gt2next = gt_spectrum[num_spectrum*360*90:]

                for i in range(num_spectrum):
                    pred_sepctrum_i = pred_spectrum[i*360*90:(i+1)*360*90].numpy().reshape(90, 360)
                    gt_spectrum_i = gt_spectrum[i*360*90:(i+1)*360*90].numpy().reshape(90, 360)
                    mae_error_i = np.mean(abs(pred_sepctrum_i - gt_spectrum_i))
                    ssim_i = ssim(pred_sepctrum_i, gt_spectrum_i, data_range=1, multichannel=False)
                    self.logger.info(f"Spectrum {save_img_idx:d}, MAE error = {mae_error_i:.6f}; SSIM = {ssim_i:.6f}")
                    paint_spectrum_compare(pred_sepctrum_i, gt_spectrum_i, save_path=os.path.join(self.log_dir, 'pred_spectrum', f'{save_img_idx}.png'))
                    all_ssim.append(ssim_i)
                    all_mae.append(mae_error_i)
                    save_img_idx += 1
            self.logger.info(
                f"SSIM: median = {np.median(all_ssim):.6f}, mean = {np.mean(all_ssim):.6f}"
            )
            self.logger.info(
                f"MAE : median = {np.median(all_mae):.6f}, mean = {np.mean(all_mae):.6f}"
            )
            np.savetxt(
                os.path.join(self.log_dir, 'all_ssim.txt'),
                np.array(all_ssim),
                fmt='%.6f'
            )
            np.savetxt(
                os.path.join(self.log_dir, 'all_mae.txt'),
                np.array(all_mae),
                fmt='%.6f'
            )


    def save_checkpoint(self):
        model_list = [x for x in sorted(os.listdir(self.checkpoints_dir)) if x.endswith('.tar')]
        if len(model_list) > 5:
            os.remove(self.checkpoints_dir + '/%s' % model_list[0])

        checkpoint_name = os.path.join(self.checkpoints_dir, f'{self.current_iter:06d}.tar')
        torch.save({
            'current_iter': self.current_iter,
            'nerf2_network_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, checkpoint_name)
        return checkpoint_name












