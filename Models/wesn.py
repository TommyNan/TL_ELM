import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ESNCell(nn.Module):
    """
    Echo State Network (ESN) cell.

    This cell implements a *fixed* reservoir with:
      - Input weights   W_in ∈ ℝ^{input_dim × hidden_dim}
      - Recurrent weights W_h ∈ ℝ^{hidden_dim × hidden_dim}
      - Optional bias      b ∈ ℝ^{hidden_dim}

    The hidden state is updated as:
        h_{t+1} = (1 - leaky) * h_t + leaky * φ( x_t W_in + h_t W_h + b )

    where φ(·) is a point-wise nonlinearity (e.g., tanh).

    All weights are non-trainable; only the output layer on top of the reservoir
    is typically trained, which is standard in ESN / reservoir computing.
    """


    def __init__(
            self,
            hidden_dim: int,
            connectivity: float = 0.1,
            leaky: float = 1.0,
            spectral_radius: float = 0.9,
            use_norm2: bool = False,
            use_bias: bool = True,
            activation: str = "tanh",
            W_in_initializer: str = "xavier_uniform",
            W_h_initializer: str = "xavier_uniform",
            bias_initializer: str = "zeros",
    ):
        """
        Args:
            hidden_dim:       Reservoir size (number of ESN neurons).
            connectivity:     Fraction of non-zero recurrent connections (sparsity of W_res).
            leaky:            Leaky integration rate in [0, 1].
            spectral_radius:  Desired spectral radius of the recurrent weight matrix.
            use_norm2:        If True, scale by L2 norm; otherwise scale by max |eigenvalue|.
            use_bias:         Whether to use additive bias.
            activation:       Nonlinearity: "tanh", "relu", or "sigmoid".
            W_in_initializer: Init for W_in.  (e.g., "xavier_uniform", "xavier_normal").
            W_h_initializer:  Init for W_h before sparsity + spectral scaling.
            bias_initializer: "zeros" or "constant".
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.connectivity = connectivity
        self.leaky = leaky
        self.spectral_radius = spectral_radius
        self.use_norm2 = use_norm2
        self.use_bias = use_bias

        # ----- activation function -----
        act_dict = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
        }
        if activation not in act_dict:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = act_dict[activation]
        self.activation_name = activation

        # ----- initializer types (as strings) -----
        self.W_in_initializer = W_in_initializer
        self.W_h_initializer = W_h_initializer
        self.bias_initializer = bias_initializer

        # Lazy-initialized parameters (depend on input_dim)
        self.W_in = None            # [input_dim, hidden_dim]
        self.W_h = None             # [hidden_dim, hidden_dim]
        self.bias = None            # [hidden_dim]

    # ------------------------------------------------------------------ #
    # Initializer helper
    # ------------------------------------------------------------------ #
    def _init_weight(
        self,
        shape: Tuple[int, int],
        init_type: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create a weight tensor with the given shape and initializer name.
        """
        w = torch.empty(*shape, device=device, dtype=dtype)

        if init_type == "xavier_uniform":
            nn.init.xavier_uniform_(w)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(w)
        elif init_type == "orthogonal":
            nn.init.orthogonal_(w)
        elif init_type == "zeros":
            nn.init.zeros_(w)
        else:
            raise ValueError(f"Unsupported initializer: {init_type}")

        return w

    # ------------------------------------------------------------------ #
    # Parameter initialization once input_dim is known
    # ------------------------------------------------------------------ #
    def _params_initialize(self, input_dim: int, device, dtype):

        # 1) Input kernel W_in: [input_dim, hidden_dim]
        W_in = self._init_weight(
            shape=(input_dim, self.hidden_dim),
            init_type=self.W_in_initializer,
            device=device,
            dtype=dtype)
        self.W_in = nn.Parameter(W_in, requires_grad=False)

        # 2) W_h kernel: shape [hidden_dim, hidden_dim]
        W_h = self._init_weight(
            shape=(self.hidden_dim, self.hidden_dim),
            init_type=self.W_h_initializer,
            device=device,
            dtype=dtype,
        )

        # Apply sparsity (connectivity mask):
        #   mask_ij ~ Bernoulli(connectivity)
        mask = (torch.rand_like(W_h) <= self.connectivity).to(dtype)
        W_h = W_h * mask

        # Scale recurrent weights to enforce the echo state property via spectral radius.
        if self.use_norm2:
            # Use matrix 2-norm
            norm2 = torch.norm(W_h, p=2)
            scaling = self.spectral_radius / (norm2 + (norm2 == 0).to(dtype))
        else:
            # Use spectral radius (maximum absolute eigenvalue)
            eigvals = torch.linalg.eigvals(W_h)          # complex eigenvalues
            max_abs_eig = eigvals.abs().max().real             # scalar
            scaling = self.spectral_radius / max_abs_eig if max_abs_eig > 0 else 1.0

        W_h = W_h * scaling
        self.W_h = nn.Parameter(W_h, requires_grad=False)  # shape: [hidden_dim, hidden_dim]

        # 3) Bias term: shape [hidden_dim]
        if self.use_bias:
            if self.bias_initializer == "zeros":
                b = torch.zeros(self.hidden_dim, device=device, dtype=dtype)
            else:
                raise ValueError(f"Unsupported initializer: {dtype}")
            self.bias = nn.Parameter(b, requires_grad=False)
        else:
            self.bias = None


    # ------------------------------------------------------------------ #
    # One ESN time-step
    # ------------------------------------------------------------------ #
    def forward(self, inputs: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        r"""
        Run one ESN time step.

        Args:
            inputs:
                Tensor of shape [B, input_dim], the current input x_t.
            prev_state:
                Tensor of shape [B, hidden_dim], the previous reservoir state h_t.

        Returns:
            new_state:
                Tensor of shape [B, hidden_dim], the updated state h_{t+1}.
        """
        device = inputs.device
        dtype = inputs.dtype

        # Lazy initialization of weights when we see the first input
        if self.W_in is None or self.W_h is None:
            input_dim = inputs.size(-1)
            self._params_initialize(input_dim, device, dtype)

        # 1) Linear term from input:  x_t @ W_in
        input_term = inputs @ self.W_in   # [B, hidden_dim]
        # 2) Linear term from previous state:  h_t @ W_h
        state_term = prev_state @ self.W_h # state_term: [B, hidden_dim]
        # 3) Combine linear terms and add bias if present
        lin_out = input_term + state_term
        if self.bias is not None:
            # bias: [hidden_dim] → broadcasted to [B, hidden_dim]
            lin_out = lin_out + self.bias

        # 4) Apply nonlinearity
        act_out = self.activation(lin_out)

        # 5) Leaky integration:
        #    h_{t+1} = (1 - leaky) * h_t + leaky * act_out
        new_state = (1.0 - self.leaky) * prev_state + self.leaky * act_out

        return new_state


class WESN(nn.Module):
    r"""
    Windowed Echo State Network (WESN).
    This module wraps an ESNCell and a linear readout:
      1) First, an ESN reservoir processes the input sequence:
           x ∈ ℝ^{B × T × C} → h ∈ ℝ^{B × T × H}
      2) Then, we optionally augment the reservoir states with a
         time-delay embedding of the inputs:
           [h_t, x_t, x_{t-1}, ..., x_{t-(win_len-1)}]
      3) Finally, a linear layer maps these features to the output dimension.

    Shapes:
        inputs:  [B, T, C]
        outputs: [B, T, output_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        connectivity: float = 0.1,
        leaky: float = 1.0,
        spectral_radius: float = 0.9,
        use_norm2: bool = False,
        use_bias: bool = True,
        activation: str = "tanh",
        win_len: int = 0,
        W_in_initializer: str = "xavier_uniform",
        W_h_initializer: str = "xavier_uniform",
        bias_initializer: str = "zeros",
        output_dim: int = 1,
        **kwargs,
    ) -> None:
        """
        Args:
            hidden_dim:        Size of the ESN reservoir (H).
            connectivity:      Fraction of non-zero entries in W_h.
            leaky:             Leaky integration coefficient in [0, 1].
            spectral_radius:   Desired spectral radius of W_h.
            use_norm2:         If True, use matrix 2-norm for scaling W_h;
                               otherwise use spectral radius.
            use_bias:          Whether ESNCell uses an additive bias.
            activation:        "tanh", "relu", or "sigmoid".
            win_len:           Length of the input time window.
                               - If 0, only reservoir states are used.
                               - If L > 0, features include:
                                   [h_t, x_t, x_{t-1}, ..., x_{t-(L-1)}]
            W_in_initializer:  Initializer for W_in.
            W_h_initializer:   Initializer for W_h.
            bias_initializer:  Initializer for ESN bias (currently "zeros").
            output_dim:        Output feature dimension per time step.
        """
        super().__init__()
        # ESN reservoir cell
        self.cell = ESNCell(
            hidden_dim=hidden_dim,
            connectivity=connectivity,
            leaky=leaky,
            spectral_radius=spectral_radius,
            use_norm2=use_norm2,
            use_bias=use_bias,
            activation=activation,
            W_in_initializer=W_in_initializer,
            W_h_initializer=W_h_initializer,
            bias_initializer=bias_initializer,
        )

        self.win_len = win_len
        self.output_dim = output_dim
        # The linear readout is created lazily once we know input_dim (C).
        # Until then we keep it as None.
        self.output_layer: Optional[nn.Linear] = None

    def _build_output_layer(
        self,
        input_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        r"""
        Create the linear readout layer.

        Feature dimension is:
            feature_dim = hidden_dim + win_len * input_dim
        because for win_len = L > 0 we concatenate:
            [h_t, x_t, x_{t-1}, ..., x_{t-(L-1)}]
        i.e., L blocks of input_dim, plus reservoir_dim.
        """
        hidden_dim = self.cell.hidden_dim
        feature_dim = hidden_dim + input_dim * self.win_len
        self.output_layer = nn.Linear(feature_dim, self.output_dim)
        self.output_layer.to(device=device, dtype=dtype)

    def forward(self, inputs):
        """
        inputs: [batch_size, time_steps, input_dim]
        Return:
        output: [batch_size, time_steps, output_dim]
        """
        # B: batch_size, T: time_steps, C: input_dim
        B, T, C = inputs.shape
        device = inputs.device
        dtype = inputs.dtype

        if self.output_layer is None:
            self._build_output_layer(input_dim=C, device=device, dtype=dtype)

        # ------- 1)  ESN reservoir -------
        # Initialize: [B, hidden_dim]
        hidden_dim = self.cell.hidden_dim
        h_t = torch.zeros(B, hidden_dim, device=device, dtype=dtype)
        reservoir_states = []

        for t in range(T):
            # inputs[:, t, :] : [B, C]
            h_t = self.cell(inputs[:, t, :], h_t)  # [B, H]
            reservoir_states.append(h_t)

        # res_out: [B, T, hidden_dim]
        res_out = torch.stack(reservoir_states, dim=1)

        # ------- 2) Construct stacked_inp（reservoir output + time-lag input）-------
        stacked_list = [res_out]  # [B, T, hidden_dim]

        if self.win_len:
            if self.win_len > T:
                raise ValueError(f"win_len ({self.win_len}) cannot be greater than input timesteps ({T})")
            # Input (lag=0)
            stacked_list.append(inputs)  # [B, T, C]

            # Other lags k=1..win_len-1
            for k in range(1, self.win_len):
                pad_front = torch.zeros(B, k, C, device=device, dtype=dtype)  # [B, k, C]
                # body: inputs shifted by k steps
                body = inputs[:, :-k, :]  # [B, T-k, C]
                # lag_k: [0, ..., 0, x_0, ..., x_{T-k-1}] ∈ ℝ^{B × T × C}
                lag_k = torch.cat([pad_front, body], dim=1)  # [B, T, C]
                stacked_list.append(lag_k)

        # stacked_inp: [B, T, feature_dim]
        stacked_inp = torch.cat(stacked_list, dim=-1)
        output = self.output_layer(stacked_inp)

        return output

if __name__ == '__main__':
    B, input_dim, hidden_dim = 4, 10, 32
    x_t = torch.randn(B, input_dim)
    h_t = torch.zeros(B, hidden_dim)

    cell = ESNCell(hidden_dim=hidden_dim)
    h_next = cell(x_t, h_t)

    print(h_next.shape)  # torch.Size([4, 32])

    # ---- Hyperparameter Setting ----
    B = 4          # batch_size
    T = 10         # time_steps
    C = 8          # input_dim
    H = 16         # hidden_dim of ESN reservoir
    out_dim = 3    # output_dim per time step
    win_len = 3    # use x_t, x_{t-1}, x_{t-2}

    # ---- Construct the Model ----
    model = WESN(
        hidden_dim=H,
        connectivity=0.1,
        leaky=1.0,
        spectral_radius=0.9,
        use_norm2=False,
        use_bias=True,
        activation="tanh",
        win_len=win_len,
        W_in_initializer="xavier_uniform",
        W_h_initializer="xavier_uniform",
        bias_initializer="zeros",
        output_dim=out_dim,
    )

    print(model)

    # ---- Random Input ----
    x = torch.randn(B, T, C)   # [B, T, C]

    # ---- Foward Propagation ----
    y = model(x)               # [B, T, out_dim]
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    # Check the dimensions of hte output
    assert y.shape == (B, T, out_dim), "Output shape mismatch!"

    # ---- Simple loss + Back-propagation ----
    loss = y.mean()
    loss.backward()

    print("\nParameter gradients:")
    for name, param in model.named_parameters():
        print(
            f"{name:25s}  requires_grad={param.requires_grad},  "
            f"has_grad={param.grad is not None}"
        )

