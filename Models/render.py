import torch
import numpy as np
import torch.nn.functional as F


def get_renderer(logger, configs):
    render_params = configs["render_params"]
    data_params = configs["data_params"]
    dataset_type = data_params["type"]

    if dataset_type == "rfid":
        renderer = SpectrumRenderer(render_params=render_params)
    elif dataset_type == "ble":
        renderer = BLERenderer(render_params=render_params)
    elif dataset_type == "mimo":
        renderer = CSIRenderer(render_params=render_params)
    logger.info(f"Loading Renderer")

    return renderer


class Renderer():
    """Ray-marching renderer that uniformly samples 3D points along rays.

    This utility is typically used in NeRF-style volumetric rendering pipelines
    to generate query points and corresponding ray parameters for subsequent
    field evaluation and numerical integration.

    Parameters
    ----------
    render_params : dict
        Dictionary containing rendering hyperparameters:
        - "num_points" (int): Number of samples per ray.
        - "near" (float): Near bound for sampling along each ray.
        - "far" (float): Far bound for sampling along each ray.
    """
    def __init__(self, render_params: dict):
        self.num_points = int(render_params["num_points"])
        self.near = float(render_params["near"])
        self.far = float(render_params["far"])

    def sample_points(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample points along rays within a fixed depth interval.


        Given ray origins ``rays_o`` and directions ``rays_d``, this function
        samples ``num_points`` distances ``t`` uniformly from ``[near, far]``,
        and constructs 3D query points via:
            ``p(t) = o + t * d``.

        The implementation is fully vectorized and supports both unbatched
        (``[N, 3]``) and batched (``[B, N, 3]``) ray tensors.

        Parameters
        ----------
        rays_o : tensor, shape [num_rays, 3] or [batch_size, num_rays(num_chunks), 3]
            Origins of rays.
        rays_d : tensor, shape [num_rays, 3] or [batch_size, num_rays(num_chunks), 3]
            Directions of rays (not necessarily normalized).

        Returns
        -------
        pts : torch.Tensor
            Sampled 3D points along each ray. Shape ``[..., num_points, 3]``.
        pts_dist : torch.Tensor
            Sampled ray parameters (distances) for each point. Shape
            ``[..., num_points]``.
        """
        device = rays_o.device
        dtype = rays_o.dtype
        # t in [near, far], uniformly sampled, then broadcast to all rays
        # pts_dist: [num_points]
        pts_dist_base = torch.linspace(self.near, self.far, steps=self.num_points, device=device, dtype=dtype)
        # pts_dist: [num_rays, num_points] or [batch_size, num_rays, num_points]
        leading_shape = rays_o.shape[:-1]  # e.g. (num_rays,) or (B, num_rays)
        view_shape = (1,) * len(leading_shape) + (self.num_points,)
        pts_dist = pts_dist_base.view(view_shape).expand(*leading_shape, self.num_points)

        # pts = o + t * d
        # rays_o: [..., 3] -> [..., 1, 3]
        # rays_d: [..., 3] -> [..., 1, 3]
        # pts:  [..., num_points, 3]
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * pts_dist.unsqueeze(-1)

        return pts, pts_dist



class SpectrumRenderer(Renderer):
    """Volumetric renderer for spectrum-aware (complex-valued) signal synthesis.

    This renderer samples 3D query points along each ray and evaluates a neural
    field model that outputs per-sample attenuation/phase parameters and a
    complex-valued source signal. The final received amplitude is obtained via
    a NeRF-style compositing rule (cumulative transmittance) combined with
    phase accumulation and a simple distance-based path-loss term.

    Notes
    -----
    - The model is assumed to return a tensor of shape ``[..., 4]`` containing
      (amp_degrade, phase_rot, sig_amp, sig_phase) per sample.
    """
    def __init__(self, render_params):
        super().__init__(render_params)

    def render(self, model, tx, rays_o, rays_d):
        """Render the received signal magnitude for each ray.

        Parameters
        ----------
        model : callable
            Neural field queried as ``model(pts, view_dirs, tx_pos)``, returning
            per-sample parameters of shape ``[num_rays, num_points, 4]``.
        tx : torch.Tensor
            Transmitter positions. Shape ``[num_rays, 3]``.
        rays_o : torch.Tensor
            Ray origins. Shape ``[num_rays, 3]``.
        rays_d : torch.Tensor
            Ray directions (not necessarily normalized). Shape ``[num_rays, 3]``.

        Returns
        -------
        receive_signal : torch.Tensor
            Rendered received signal magnitude per ray. Shape ``[num_rays]``.
        """

        device = rays_o.device
        dtype = rays_o.dtype

        # ---------------------------------------------------------------------
        # 1) Sample query points along each ray: pts = o + t * d
        # ---------------------------------------------------------------------
        pts, pts_dist = self.sample_points(rays_o, rays_d)  # pts: [num_rays, num_points, 3]  pts_dist: [num_rays, num_points]
        num_rays, num_points = pts.shape[:2]

        # Broadcast view directions and transmitter positions to per-sample shape.
        view_expanded = rays_d.unsqueeze(1).expand(-1, num_points, -1)    # [num_rays, num_points, 3]
        tx_expanded = tx.unsqueeze(1).expand(-1, num_points, -1)  # [num_rays, num_points, 3]

        attn_and_sig = model(pts, view_expanded, tx_expanded) # [num_rays, num_points, 4]
        amp_degrade = attn_and_sig[..., 0]  # [num_rays,num_points]
        phase_rot = attn_and_sig[..., 1]  # [num_rays,num_points]
        sig_amp = attn_and_sig[..., 2]  # [num_rays,num_points]
        sig_phase = attn_and_sig[..., 3]  # [num_rays,num_points]

        # ---------------------------------------------------------------------
        # 2) Convert t-samples into physical step sizes along the ray.
        #    If rays_d is not normalized, scale by ||d||.
        # ---------------------------------------------------------------------
        # Δt: [R, P], last step is a large sentinel to emulate "infinite" distance.
        dt = pts_dist[..., 1:] - pts_dist[..., :-1]  # [num_rays, num_points-1]
        dt_last = torch.full((num_rays, 1), 1e10, device=device, dtype=dtype)
        dt = torch.cat([dt, dt_last], -1)  # [num_rays, num_points]
        ray_lengths = torch.norm(rays_d, dim=-1, keepdim=True)  # [num_rays, 1]
        dists = dt * ray_lengths    # [num_rays, num_points]

        # ---------------------------------------------------------------------
        # 3) Enforce parameter ranges.
        #    - amplitude terms: non-negative
        #    - phase terms: [0, 2π]
        # ---------------------------------------------------------------------
        # Amplitude >=0
        amp_degrade = abs(F.leaky_relu(amp_degrade))    # [num_rays, num_points]
        sig_amp = abs(F.leaky_relu(sig_amp))  # [num_rays, num_points]
        # amp_degrade = F.softplus(amp_degrade)  # smoother than abs(leaky_relu)
        # sig_amp = F.softplus(sig_amp)

        # Phase [0, 2π]
        phase_rot = 2*np.pi*torch.sigmoid(phase_rot)    # [num_rays, num_points]
        sig_phase = 2*np.pi*torch.sigmoid(sig_phase)    # [num_rays, num_points]

        # ---------------------------------------------------------------------
        # 4) NeRF-style compositing with phase accumulation.
        #    alpha_i = 1 - exp(-sigma_i * Δs_i)
        #    T_i     = Π_{k<i} (1 - alpha_k)
        #    φ_i     = exp(j * Σ_{k<=i} Δφ_k)
        # ---------------------------------------------------------------------
        # alpha_i = 1 - exp(-amp_degrade_i * d_i)
        alpha = 1 - torch.exp(-amp_degrade * dists)  # [num_rays, num_points]
        phase = phase_rot * dists   # [num_rays, num_points]

        # T_i = prod_{k < i} (1 - alpha_k + eps)
        eps = 1e-10
        att_i = torch.cumprod(torch.cat([torch.ones((num_rays, 1)).cuda(), 1.0 - alpha + eps], -1), -1)[:, :-1]   # [num_rays, num_points]

        # φ_i = sum_{k <= i} Δφ_k
        # phase_i = torch.cumsum(torch.cat([torch.ones((num_rays, 1)).cuda(), phase], -1), -1)[:, :-1]
        phase_i = torch.cumsum(torch.cat([torch.zeros((num_rays, 1)).cuda(), phase], -1), -1)[:, :-1]
        # phase_i = torch.cumsum(phase, dim=-1)
        phase_i = torch.exp(1j*phase_i)    # [num_rays, num_points]

        path = torch.cat([pts_dist[..., 1:], dt_last], -1)
        path_loss = 0.025 / path

        sig_complex = sig_amp * torch.exp(1j * sig_phase)

        receive_signal = torch.sum(sig_complex*att_i*phase_i*path_loss, -1)  # [num_rays]
        receive_signal = abs(receive_signal)

        return receive_signal


class BLERenderer(Renderer):
    def __init__(self, render_params):
        super().__init__(render_params)

    def render(self, model, tx, rays_o, rays_d):
        """render the signal strength of each ray

        Parameters
        ----------
        tx: tensor. [batch_size, 3]. The position of the transmitter
        rays_o : tensor. [batch_size, 3]. The origin of rays
        rays_d : tensor. [batch_size, 9x36x3]. The direction of rays

        Return
        ----------
         recv_signal : tensor, shape [batch_size]
         Aggregated received signal over all ray chunks.
        """
        device = tx.device
        dtype = tx.dtype
        batch_size = tx.shape[0]

        # ---- reshape rays_d to [B, N_rays, 3] ----
        rays_d = rays_d.view(batch_size, -1, 3)  # [B, N_rays, 3]
        total_rays = rays_d.shape[1]  # N_rays per batch item

        # ---- chunk settings ----
        chunk_size = 36  # number of rays per chunk
        # num_chunks = math.ceil(total_rays / chunk_size)  # how many chunks we need
        # num_chunks = 36 // chunk_size
        num_chunks = 36 * 9 // chunk_size

        # rays_o_expanded: [B, N_rays, 3]
        rays_o_expanded = rays_o.unsqueeze(1).expand(-1, chunk_size, -1)
        # tx_expanded: [B, N_rays, 3]
        tx_expanded = tx.unsqueeze(1).expand_as(rays_o_expanded)

        # ---- accumulator for received signal per batch ----
        recv_signal = torch.zeros(batch_size, device=device)

        # ---- process rays in chunks ----
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_rays)

            # Each chunk: [B, C, 3]
            rays_d_chunk = rays_d[:, start:end, :]  # [B, C, 3]
            rays_o_chunk = rays_o_expanded[:, start:end, :]  # [B, C, 3]
            tx_chunk = tx_expanded[:, start:end, :]  # [B, C, 3]

            # Sample points along rays
            # pts: [B, C, num_points, 3]
            # pts_dist: [B, C, num_points]
            pts, pts_dist = self.sample_points(rays_o_chunk, rays_d_chunk)

            # views_chunk: same shape as pts, direction broadcast到所有采样点
            # [B, C, num_points, 3]
            views_chunk = rays_d_chunk.unsqueeze(2).expand_as(pts)
            tx_chunk_expanded = tx_chunk.unsqueeze(2).expand_as(pts)  # [B, C, num_points, 3]

            # Run network and compute outputs for this chunk
            # attn_and_sig: [B, C, num_points, 4]
            attn_and_sig = model(pts, views_chunk, tx_chunk_expanded)

            wavelength = 3.e8 / 2.4e9

            # the dist interval of each point (equal)
            dists = pts_dist[..., 1:] - pts_dist[..., :-1]  # [batch_size, C, num_points-1]
            last_value = torch.full((batch_size, chunk_size, 1), 1e10, device=device, dtype=dtype)
            dists = torch.cat([dists, last_value], -1)  # [batch_size, C, num_points]
            ray_lengths = torch.norm(rays_d, dim=-1, keepdim=True)  # [batch_size, C, 1]
            dists = dists * ray_lengths  # [batch_size, C, num_points]

            amp_degrade, phase_rot, sig_amp, sig_phase = (attn_and_sig[..., 0], attn_and_sig[..., 1],
                                                          attn_and_sig[..., 2], attn_and_sig[..., 3])  # [B, C, num_points]

            # Amplitude >=0
            amp_degrade = abs(F.leaky_relu(amp_degrade))  # [B, C, num_points]
            sig_amp = abs(F.leaky_relu(sig_amp))  # [B, C, num_points]

            # Phase [-π, π]
            phase_rot = 2 * np.pi * torch.sigmoid(phase_rot) - np.pi  # [B, C, num_points]
            sig_phase = 2 * np.pi * torch.sigmoid(sig_phase) -np.pi # [B, C, num_points]

            alpha = -amp_degrade * dists  # [B, C, num_points]
            phase = phase_rot + 2*np.pi*dists/wavelength  # [B, C, num_points]

            # T_i = prod_{k < i} (1 - alpha_k + eps)
            att_i = torch.exp(torch.cumsum(alpha, -1))
            phase_i = torch.exp(1j*torch.cumsum(phase, -1))
            sig_complex = sig_amp * torch.exp(1j * sig_phase)

            receive_signal_chunk = torch.sum(sig_complex * att_i * phase_i, -1)  # [batch_size, chunk_size]
            receive_signal_chunk = torch.sum(receive_signal_chunk, -1)  # [batch_size]
            receive_signal_chunk = abs(receive_signal_chunk)    # [batch_size]

            # accumulate per batch
            recv_signal += receive_signal_chunk

        return recv_signal  # [batch_size]


class CSIRenderer(Renderer):
    def __init__(self, render_params):
        super().__init__(render_params)

    def render(self, model, uplink, rays_o, rays_d):
        """render the signal strength of each ray

        Parameters
        ----------
        uplink: tensor. [batch_size, 52]. The Uplink CSI
        rays_o : tensor. [batch_size, 3]. The origin of rays
        rays_d : tensor. [batch_size, 9x36x3]. The direction of rays

        Return
        ----------
         recv_signal : tensor, shape [batch_size]
         Aggregated received signal over all ray chunks.
        """
        device = uplink.device
        dtype = uplink.dtype
        batch_size = uplink.shape[0]

        # ---- reshape rays_d to [B, N_rays, 3] ----
        rays_d = rays_d.view(batch_size, -1, 3)  # [batch_size, num_rays, 3]
        total_rays = rays_d.shape[1]  # N_rays per batch item

        # rays_o_expanded: [batch_size, num_rays, 3]
        rays_o_expanded = rays_o.unsqueeze(1).expand(-1, total_rays, -1)
        # uplink_expanded: [batch_size, num_rays, 52]
        uplink_expanded = uplink.unsqueeze(1).expand(-1, total_rays, -1)

        pts, pts_dist = self.sample_points(rays_o_expanded, rays_d) # [batch_size, 9x36, num_points, 3] [batch_size, 9x36, num_points]

        # views: same shape as pts, direction broadcast到所有采样点
        # [B, C, num_points, 3]
        views = rays_d.unsqueeze(2).expand(-1, -1, self.num_points, -1)
        uplink_expanded = uplink_expanded.unsqueeze(2).expand(-1, -1, self.num_points, -1)  # [B, C, num_points, 3]

        attn_and_sig = model(pts, views, uplink_expanded)

        wavelength = 3.e8 / 2.4e9

        # the dist interval of each point (equal)
        dists = pts_dist[..., 1:] - pts_dist[..., :-1]  # [batch_size, 36*9, num_points-1]
        last_value = torch.full((batch_size, total_rays, 1), 1e10, device=device, dtype=dtype)
        dists = torch.cat([dists, last_value], -1)  # [batch_size, num_rays, num_points]
        ray_lengths = torch.norm(rays_d, dim=-1, keepdim=True)  # [batch_size, num_rays, 1]
        dists = dists * ray_lengths  # [batch_size, num_rays, num_points]

        amp_degrade, phase_rot, sig_amp, sig_phase = (attn_and_sig[..., :26], attn_and_sig[..., 26:52],
                                                      attn_and_sig[..., 52:78],
                                                      attn_and_sig[..., 78:104])  # [B, C, num_points]

        # Amplitude >=0
        amp_degrade = abs(F.leaky_relu(amp_degrade))  # [B, num_rays, num_points, 26]
        sig_amp = abs(F.leaky_relu(sig_amp))  # [B, num_rays, num_points, 26]

        # Phase [-π, π]
        phase_rot = 2 * np.pi * torch.sigmoid(phase_rot) - np.pi  # [B, num_rays, num_points, 26]
        sig_phase = 2 * np.pi * torch.sigmoid(sig_phase) - np.pi  # [B, num_rays, num_points, 26]

        dists = dists.unsqueeze(-1)
        alpha = -amp_degrade * dists  # [B, num_rays, num_points, 26]
        phase = phase_rot + 2 * np.pi * dists / wavelength  # [B, num_rays, num_points, 26]

        # T_i = prod_{k < i} (1 - alpha_k + eps)
        att_i = torch.exp(torch.cumsum(alpha, -2))  # [B, num_rays, num_points, 26]
        phase_i = torch.exp(1j * torch.cumsum(phase, -2))   # [B, num_rays, num_points, 26]

        sig_complex = sig_amp * torch.exp(1j * sig_phase)  # [B, num_rays, num_points, 26]
        receive_signal = torch.sum(sig_complex * att_i * phase_i, -2)  # [B, num_rays, 26]
        receive_signal = torch.sum(receive_signal, 1)  # [batch_size, 26]

        return receive_signal # [batch_size, 26]






