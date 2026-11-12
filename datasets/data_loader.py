import os.path
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation
import imageio
import torch
import yaml
from tqdm import tqdm
from einops import rearrange

# from dataloader import Spectrum_dataset, CSI_dataset, BLE_dataset

def rssi2amplitude(rssi):
    """convert rssi to amplitude
    """
    return 1 - (rssi / -100)


def amplitude2rssi(amplitude):
    """convert amplitude to rssi
    """
    return -100 * (1 - amplitude)

def generate_train_test_index(dataset_type, data_dir, ratio):
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(data_dir, "spectrum")
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
    elif dataset_type == "ble":
        rssi_dir = os.path.join(data_dir, "gateway_rssi.csv")
        index = pd.read_csv(rssi_dir).index.values
    elif dataset_type == "mimo":
        mimo_dir = os.path.join(data_dir, "csidata.npy")
        index = [i for i in range(np.load(mimo_dir).shape[0])]

    random.shuffle(index)
    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])[:1000]
    test_index = np.array(index[train_len:])[:500]

    np.savetxt(os.path.join(data_dir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(data_dir, "test_index.txt"), test_index, fmt='%s')

def get_dataloaders(logger, configs):
    data_params = configs["data_params"]
    train_params = configs["train_params"]
    dataset_type = data_params["type"]
    data_dir = data_params["data_dir"]
    scale_worldsize = data_params["scale_worldsize"]
    if dataset_type in ["rfid", "ble", "mimo"]:
        logger.info("Generating dataset ")
    # if dataset_type == "rfid":
    #     dataset = Spectrum_dataset
    # elif dataset_type == "mimo":
    #     dataset = CSI_dataset
    # elif dataset_type == "ble":
    #     dataset = BLE_dataset
    # else:
    #     raise ValueError("Unknown dataset type")

    train_index = os.path.join(data_dir, "train_index.txt")
    test_index = os.path.join(data_dir, "test_index.txt")

    if not os.path.exists(train_index) or not os.path.exists(test_index):
        logger.info(f"The train and test index files do not exist for the {dataset_type}, generate them")
        generate_train_test_index(dataset_type, data_dir, data_params["train_ratio"])
        logger.info(f"Successfully generate the train and test index files for the {dataset_type}")


    if dataset_type == "rfid":
        train_set = SpectrumDataset(data_dir, train_index, scale_worldsize)
        test_set = SpectrumDataset(data_dir, test_index, scale_worldsize)
    elif dataset_type == "ble":
        train_set = BLEDataset(data_dir, train_index, scale_worldsize)
        test_set = BLEDataset(data_dir, test_index, scale_worldsize)
    elif dataset_type == "mimo":
        train_set = CSIDataset(data_dir, train_index, scale_worldsize)
        test_set = CSIDataset(data_dir, test_index, scale_worldsize)
    logger.info(f"Loading training set and test set [{dataset_type}]... No. of raw samples: Train[{len(train_set)}]|Test[{len(test_set)}]")

    train_iter = DataLoader(train_set, train_params["batch_size"], shuffle=True, num_workers=0)
    test_iter = DataLoader(test_set, train_params["batch_size"], shuffle=False, num_workers=0)

    return train_iter, test_iter

class SpectrumDataset(Dataset):
    def __init__(self, data_dir, index_dir, scale_worldsize=1):
        super(SpectrumDataset, self).__init__()
        self.data_dir = data_dir
        self.tx_pos_dir = os.path.join(data_dir, 'tx_pos.csv')  # [6123, 3]
        self.gateway_pos_dir = os.path.join(data_dir, 'gateway_info.yml')   # Fixed
        self.spectrum_dir = os.path.join(data_dir, 'spectrum')
        self.scale_worldsize = float(scale_worldsize)

        self.spectrum_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])
        # Load resolution
        spectrum_example = imageio.imread(os.path.join(self.spectrum_dir, self.spectrum_names[0]))
        self.num_elevation, self.num_azimuth = spectrum_example.shape
        self.rays_per_spectrum = self.num_elevation * self.num_azimuth

        self.dataset_index = np.loadtxt(index_dir, dtype=str)

        self._load_env_info()

        self.nn_inputs, self.nn_labels = self.load_data()

    def __len__(self):
        return len(self.dataset_index) * self.rays_per_spectrum

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]


    # ---------------- internal helpers ---------------- #
    """Load gateway, tx_pos, generate spectrum-shared ray_o / ray_d。"""
    def _load_env_info(self):
        # 1) gateway position and orientation
        with open(self.gateway_pos_dir, "r") as f:
            gateway_info = yaml.safe_load(f)

        gateway_pos = np.asarray(gateway_info["gateway1"]["position"], dtype=np.float32)
        gateway_orientation = np.asarray(gateway_info["gateway1"]["orientation"], dtype=np.float32)  # Quaternion [x, y, z, w]

        # Optional: scale the range of world coordinates
        gateway_pos *= self.scale_worldsize
        self.gateway_pos = torch.from_numpy(gateway_pos)  # [3]

        # 2) Tx pos
        tx_pos_np = pd.read_csv(self.tx_pos_dir).values.astype(np.float32)  # [N_tx, 3]
        tx_pos_np *= self.scale_worldsize
        self.tx_pos = torch.from_numpy(tx_pos_np)  # [N_tx, 3]

        # 3) Predefine the ray orientation and direction
        self.ray_o, self.ray_d = self.gen_rays_spectrum(self.gateway_pos, gateway_orientation)  # [n_rays, 3], [n_rays, 3]


    def load_data(self):
        """Expand spectrum to Input: [N_spectrum * n_rays, 9]和 Label:[.,1]。

        Returns
        -------
        nn_inputs : tensor, shape [N_samples, 9]
            The input of each ray (ray_o, ray_d, tx_pos)
        nn_labels : tensor, shape [N_samples, 1]
            The label of each pixel
        """
        num_spectrum = len(self.dataset_index)
        num_rays = self.rays_per_spectrum
        num_samples = num_spectrum * num_rays

        ## NOTE! Each spectrum will cost 1.2 MB of memory. Large dataset may cause OOM?
        nn_inputs = torch.empty((num_samples, 9), dtype=torch.float32)
        nn_labels = torch.empty((num_samples, 1), dtype=torch.float32)

        ray_o = self.ray_o  # [n_rays, 3]
        ray_d = self.ray_d  # [n_rays, 3]

        for i, idx in tqdm(enumerate(self.dataset_index), total=num_spectrum, desc="Loading spectrum data"):
            # 1) Load spectrum dataset -> [num_elevation, num_azimuth]
            img_path = os.path.join(self.spectrum_dir, f"{idx}.png")
            spectrum_np = imageio.imread(img_path).astype(np.float32) / 255.0
            spectrum = torch.from_numpy(spectrum_np).view(-1, 1)  # [n_rays, 1]

            # 2) Find corresponding tx_pos
            idx_int = int(idx)
            tx = self.tx_pos[idx_int - 1]  # [3]
            tx_pos_i = tx.unsqueeze(0).expand(num_rays, 3)  # [n_rays, 3]

            # 3)
            start = i * num_rays
            end = start + num_rays

            nn_inputs[start:end, :] = torch.cat([ray_o, ray_d, tx_pos_i], dim=1)
            nn_labels[start:end, :] = spectrum

        return nn_inputs, nn_labels

    def gen_rays_spectrum(self, gateway_pos, gateway_orientation):
        """generate sample rays origin at gateway with resolution given by spectrum

        Parameters
        ----------
        azimuth : int. The number of azimuth angles
        elevation : int. The number of elevation angles

        Returns
        -------
        r_o : tensor. [n_rays, 3]. The origin of rays
        r_d : tensor. [n_rays, 3]. The direction of rays, unit vector
        """

        azimuth = torch.linspace(1, 360, self.num_azimuth) / 180 * np.pi
        elevation = torch.linspace(1, 90, self.num_elevation) / 180 * np.pi

        azimuth = azimuth.repeat(self.num_elevation)  # [1,2,3...360,1,2,3...360,...] pytorch 2.0
        elevation = elevation.repeat_interleave(self.num_azimuth)  # [1,1,1,...,2,2,2,...,90,90,90,...]

        # Local coordinates
        x = torch.cos(elevation) * torch.cos(azimuth) # [n_azi * n_ele], i.e., [n_rays]
        y = torch.cos(elevation) * torch.sin(azimuth)
        z = torch.sin(elevation)

        # Construct Rotation matrix from quant and construct World coordinates
        r_d_local = torch.stack([x, y, z], dim=0)  # [3, n_rays] 3D direction of rays in gateway coordinate
        R = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()
        r_d_world = (R @ r_d_local).T  # 3D direction of rays in world coordinate: [n_rays, 3]

        r_o = gateway_pos.unsqueeze(0).expand(self.rays_per_spectrum, 3)  # [n_rays, 3]

        return r_o, r_d_world


class BLEDataset(Dataset):
    """BLE dataset class.

    This dataset builds per-(TX, gateway) samples:
      input:  [tx_pos (3), gateway_pos (3), all gateway ray directions (beta_res * alpha_res * 3)]
      label:  RSSI (later converted to amplitude)
    """
    def __init__(self, data_dir, index_dir, scale_worldsize: float = 1.0) -> None:
        super().__init__()
        self.data_dir = data_dir
        tx_pos_dir = os.path.join(data_dir, "tx_pos.csv")
        self.gateway_pos_dir = os.path.join(data_dir, "gateway_position.yml")
        self.rssi_dir = os.path.join(data_dir, "gateway_rssi.csv")
        self.beta_res, self.alpha_res = 9, 36  # resolution of rays: betas x alphas
        self.dataset_index = np.loadtxt(index_dir, dtype=int)  # numpy array of TX indices

        # ---------------------------------------------------------------------
        # Load gateway positions
        # ---------------------------------------------------------------------
        with open(self.gateway_pos_dir, "r") as f:
            gateway_pos_dict = yaml.safe_load(f)

        # gateway_pos_dict is something like: {gw0: [x,y,z], gw1: [x,y,z], ...}
        gateway_pos_list = [pos for pos in gateway_pos_dict.values()]
        gateway_pos = np.asarray(gateway_pos_list, dtype=np.float32)
        gateway_pos /= float(scale_worldsize)
        self.gateway_pos = torch.from_numpy(gateway_pos)  # [num_gateways, 3]
        self.num_gateways = self.gateway_pos.shape[0]
        # ---------------------------------------------------------------------
        # Load transmitter positions
        # ---------------------------------------------------------------------
        tx_poses_np = pd.read_csv(tx_pos_dir).values.astype(np.float32)
        tx_poses_np /= float(scale_worldsize)
        self.tx_poses = torch.from_numpy(tx_poses_np)  # [N_tx, 3]

        # ---------------------------------------------------------------------
        # Load gateway received RSSI
        #   rssis: [N_tx, n_gateways]
        # ---------------------------------------------------------------------
        rssi_np = pd.read_csv(self.rssi_dir).values.astype(np.float32)
        self.rssis = torch.from_numpy(rssi_np)

        # ---------------------------------------------------------------------
        # Precompute rays for all gateways
        # ---------------------------------------------------------------------
        gateways_ray_o, gateways_rays_d = self.gen_rays_gateways()
        # gateways_ray_o: [n_gateways, 1, 3]
        # gateways_rays_d: [n_gateways, beta_res * alpha_res, 3]

        # Also keep flattened versions to avoid repeated .view(-1) in the loop
        self.gateways_ray_o_flat = gateways_ray_o.view(self.num_gateways, -1)          # [n_gateways, 3]
        self.gateways_rays_d_flat = gateways_rays_d.view(self.num_gateways, -1)        # [n_gateways, beta_res * alpha_res * 3]

        # ---------------------------------------------------------------------
        # Precompute total number of valid samples (RSSI != -100)
        # ---------------------------------------------------------------------
        rssi_selected = self.rssis[self.dataset_index]        # [len(index), n_gateways]
        valid_mask = (rssi_selected != -100)
        self.num_samples = int(valid_mask.sum().item())

        # Build all nn_inputs / nn_labels into memory
        self.nn_inputs, self.nn_labels = self.load_data()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int):
        return self.nn_inputs[index], self.nn_labels[index]

    # -------------------------------------------------------------------------
    # Data building
    # -------------------------------------------------------------------------
    def load_data(self):
        """Load data into memory for training.

        Returns
        -------
        nn_inputs : tensor, shape [n_samples, 3 + 3 + 3 * alpha_res * beta_res]
            Input features:
              [tx_pos (3),
               gateway_pos (3),
               flattened gateway ray directions (beta_res * alpha_res * 3)]
        nn_labels : tensor, shape [n_samples, 1]
            RSSI labels converted to amplitude.
        """
        feature_dim = 3 + 3 + 3 * self.alpha_res * self.beta_res
        nn_inputs = torch.empty((self.num_samples, feature_dim), dtype=torch.float32)
        nn_labels = torch.empty((self.num_samples, 1), dtype=torch.float32)

        data_counter = 0

        # Iterate over all selected TX indices
        for tx_idx in tqdm(self.dataset_index, total=len(self.dataset_index), desc="Loading BLE data"):
            # tx_idx is an integer index into tx_poses / rssi rows
            rssis = self.rssis[tx_idx]              # [n_gateways]
            tx_pos = self.tx_poses[tx_idx].view(-1) # [3]

            for gw_idx, rssi in enumerate(rssis):
                # Only keep valid RSSI values
                if rssi != -100:
                    # Flattened gateway position and directions
                    gateway_ray_o_flat = self.gateways_ray_o_flat[gw_idx]  # [3]
                    gateway_rays_d_flat = self.gateways_rays_d_flat[gw_idx]  # [beta_res * alpha_res * 3]

                    # Concatenate all features into one 1D vector
                    nn_inputs[data_counter] = torch.cat([tx_pos, gateway_ray_o_flat, gateway_rays_d_flat], dim=-1)
                    nn_labels[data_counter] = rssi
                    data_counter += 1

        # Convert RSSI to amplitude
        nn_labels = rssi2amplitude(nn_labels)

        return nn_inputs, nn_labels

    # -------------------------------------------------------------------------
    # Ray generation
    # -------------------------------------------------------------------------
    def gen_rays_gateways(self):
        """Generate sample rays originating at each gateway.

        For each gateway, we sample (beta_res x alpha_res) rays
        with different azimuth (alpha) and elevation (beta) angles.

        Returns
        -------
        r_o : tensor, shape [num_gateways, 1, 3]
            Ray origins (gateway positions in world coordinates).
        r_d : tensor, shape [num_gateways, beta_res * alpha_res, 3]
            Ray directions, unit vectors in world coordinates.
        """
        # Alpha: azimuth angles [0, 10, ..., 350] degrees
        alphas = torch.linspace(0.0, 350.0, self.alpha_res) / 180.0 * np.pi
        # Beta: elevation angles [10, ..., 90] degrees
        betas = torch.linspace(10.0, 90.0, self.beta_res) / 180.0 * np.pi

        # Create a grid of angles:
        #  - alphas repeated for each beta
        #  - betas repeated_interleave for each alpha
        alphas = alphas.repeat(self.beta_res)                     # [beta_res * alpha_res]
        betas = betas.repeat_interleave(self.alpha_res)           # [beta_res * alpha_res]

        radius = 1.0
        x = radius * torch.cos(alphas) * torch.cos(betas)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        # Directions per gateway (local directions are same for all gateways)
        r_d = torch.stack([x, y, z], dim=-1)  # [beta_res * alpha_res, 3]
        # Expand to all gateways
        r_d = r_d.unsqueeze(0).expand(self.num_gateways, -1, -1).contiguous()   # [num_gateways, beta_res * alpha_res, 3]
        # Origins: each gateway position, broadcast along ray dimension
        r_o = self.gateway_pos.unsqueeze(1).contiguous()  # [num_gateways (21), 1, 3]

        return r_o, r_d

class CSIDataset(Dataset):
    """
    CSI dataset class.
    """

    def __init__(self, data_dir, index_dir, scale_worldsize: float = 1.0) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.scale_worldsize = float(scale_worldsize)
        self.csi_path = os.path.join(data_dir, "csidata.npy")   # e.g. complex CSI saved as npy
        self.bs_pos_path = os.path.join(data_dir, "base-station.yml")
        self.beta_res, self.alpha_res = 9, 36  # resulution of rays

        # ------------------------------------------------------------------
        # Load indices for this split (training / validation etc.)
        # ------------------------------------------------------------------
        self.dataset_index = np.loadtxt(index_dir, dtype=int)  # numpy array of snapshot indices

        # ------------------------------------------------------------------
        # Load BS information shape: [8, 3]
        # ------------------------------------------------------------------
        with open(self.bs_pos_path, "r") as f:
            bs_pos_dict = yaml.safe_load(f)

        bs_pos = torch.as_tensor(bs_pos_dict["base_station"], dtype=torch.float32)  # [8, 3]
        bs_pos = bs_pos / scale_worldsize
        self.bs_pos = bs_pos  # [8, 3]
        self.num_bs = len(self.bs_pos)

        # ------------------------------------------------------------------
        # Load CSI data
        # ------------------------------------------------------------------
        # csi_data: [N, 8, 52] complex values
        csi_np = np.load(self.csi_path)                      # numpy array, complex
        csi_data = torch.from_numpy(csi_np)                     # [4000, 8, 52], complex
        csi_data = self.normalize_csi(csi_data)                 # keep same format

        # Split uplink / downlink on subcarrier dimension
        uplink, downlink = csi_data[..., :26], csi_data[..., 26:]  # [N, 8, 26] each
        # Split into real and imag parts
        up_real, up_imag = torch.real(uplink), torch.imag(uplink)
        down_real, down_imag = torch.real(downlink), torch.imag(downlink)

        # Concatenate real/imag along the last dim → [N, 8, 52]
        self.uplink = torch.cat([up_real, up_imag], dim=-1)       # [N, 8, 52]
        self.downlink = torch.cat([down_real, down_imag], dim=-1) # [N, 8, 52]

        # Rearrange group dimension into batch: [N*8, 52]
        self.uplink = self.uplink.reshape(-1, self.uplink.size(-1))  # [N*8, 52]
        self.downlink = self.downlink.reshape(-1, self.downlink.size(-1))

        # Build training tensors
        self.nn_inputs, self.nn_labels = self.load_data()

    def normalize_csi(self, csi):
        self.csi_max = torch.max(abs(csi))
        return csi / self.csi_max

    def denormalize_csi(self, csi):
        assert self.csi_max is not None, "Please normalize csi first"
        return csi * self.csi_max

    # ----------------------------------------------------------------------
    # Core dataset API
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.dataset_index) * self.num_bs

    def __getitem__(self, index: int):
        return self.nn_inputs[index], self.nn_labels[index]

    # ----------------------------------------------------------------------
    # Data building
    # ----------------------------------------------------------------------
    def load_data(self):
        """Load data from datadir to memory for training.

        Returns
        -------
        nn_inputs : tensor, shape [n_samples, 1027]
            Inputs for training:
              uplink: 52 (26 real + 26 imag),
              ray_o: 3,
              ray_d: 9 x 36 x 3,
              n_samples = n_dataset * n_bs
        nn_labels : tensor, shape [n_samples, 52]
            Downlink channels as labels.
        """

        # Total number of samples = (#indices) * n_bs
        num_dataset = len(self.dataset_index)
        num_samples = num_dataset * self.num_bs

        feat_dim = 52 + 3 + 3 * self.alpha_res * self.beta_res  # 52 + 3 + 3*36*9 = 1027

        # Allocate tensors directly on CPU
        nn_inputs = torch.empty((num_samples, feat_dim), dtype=torch.float32)
        nn_labels = torch.empty((num_samples, 52), dtype=torch.float32)

        bs_ray_o, bs_rays_d = self.gen_rays_gateways()

        # Flatten:
        #   [n_bs, 1, 3]        -> [n_bs, 3]
        #   [n_bs, n_rays, 3]   -> [n_bs, n_rays * 3]
        bs_ray_o_flat = bs_ray_o.view(self.num_bs, -1)            # [n_bs, 3]
        bs_rays_d_flat = bs_rays_d.view(self.num_bs, -1)          # [n_bs, n_rays * 3]

        # ------------------------------------------------------------------
        # Fill nn_inputs / nn_labels
        # ------------------------------------------------------------------
        for data_counter, idx in tqdm(enumerate(self.dataset_index), total=num_dataset, desc="Loading CSI data"):
            # For each dataset index, we have n_bs rows in uplink/downlink
            # uplink / downlink shapes: [N*8, 52], and n_bs is typically 8
            start_bs = idx * self.num_bs
            end_bs = (idx + 1) * self.num_bs

            bs_uplink = self.uplink[start_bs:end_bs]      # [num_bs, 52]
            bs_downlink = self.downlink[start_bs:end_bs]  # [num_bs, 52]

            # Global sample index range in nn_inputs / nn_labels
            out_start = data_counter * self.num_bs
            out_end = out_start + self.num_bs

            # Concatenate features: [n_bs, 52 + 3 + n_rays*3]
            nn_inputs[out_start:out_end] = torch.cat([bs_uplink, bs_ray_o_flat, bs_rays_d_flat], dim=-1)
            nn_labels[out_start:out_end] = bs_downlink

        return nn_inputs, nn_labels

    def gen_rays_gateways(self):
        """Generate sample rays originating at base stations.

        For each base station, we sample (alpha_res x beta_res) rays.

        Returns
        -------
        r_o : tensor, shape [n_bs, 1, 3]
            Ray origins (BS positions)
        r_d : tensor, shape [n_bs, n_rays, 3]
            Ray directions, unit vectors.
        """
        # Alpha: azimuth angles [0, 10, ..., 350] degrees
        alphas = torch.linspace(0.0, 350.0, self.alpha_res) / 180.0 * np.pi
        # Beta: elevation angles [10, ..., 90] degrees
        betas = torch.linspace(10.0, 90.0, self.beta_res) / 180.0 * np.pi

        # Create angle grid
        #   alphas: [beta_res * alpha_res]
        #   betas:  [beta_res * alpha_res]
        alphas = alphas.repeat(self.beta_res)                 # [0,1,2,...] length = beta_res*alpha_res
        betas = betas.repeat_interleave(self.alpha_res)       # [0,0,0,...] length = beta_res*alpha_res

        radius = 1.0
        x = radius * torch.cos(alphas) * torch.cos(betas)
        y = radius * torch.sin(alphas) * torch.cos(betas)
        z = radius * torch.sin(betas)

        # Single-BS directions: [n_rays, 3]
        r_d_single = torch.stack([x, y, z], dim=-1)           # [beta_res * alpha_res, 3]

        # Expand to all BS: [n_bs, n_rays, 3]
        r_d = r_d_single.unsqueeze(0).expand(self.num_bs, -1, -1).contiguous()

        # Origins: BS positions, broadcast along ray dimension
        # self.bs_pos: [n_bs, 3] or [3] with n_bs=1
        # Multiple BS: [n_bs, 3] -> [n_bs, 1, 3]
        r_o = self.bs_pos.unsqueeze(1).contiguous()       # [n_bs, 1, 3]

        return r_o, r_d

    def __getitem__(self, index):
        return self.nn_inputs[index], self.nn_labels[index]

    def __len__(self):
        return len(self.dataset_index) * self.num_bs