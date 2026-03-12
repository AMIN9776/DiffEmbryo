"""
diffusion_model.py

Two components:

  1. DiffusionSchedule
     Linear beta schedule.  Computes √ᾱ_s and √(1-ᾱ_s) for every step so
     we can corrupt Y_0 → Y_s in a single shot (the "q-sample" trick).

     Valid frames  : full noise according to the schedule
     Padding frames: fixed tiny noise (sigma_pad ≪ 1) so the model sees a
                     near-clean signal there and learns to ignore those positions
                     — they exist only to make variable-length batching possible.

  2. DiffusionTransformer
     A standard Transformer encoder that acts as the denoising network.

     Input  : Y_s  (B, 550, 18) noisy matrix  +  step index s
     Output : Ŷ_0  (B, 550, 18) predicted clean logits

     Key design decisions
     ─────────────────────
     • Padding mask  — starting/ending positions are passed as
       src_key_padding_mask=True so they are fully excluded from all
       attention computations.  Valid frames never attend to them, and
       they never influence the representation of valid frames.

     • Step embedding  — sinusoidal encoding of s → small MLP → d_model.
       Added to every token before the transformer so the network knows
       how much noise to expect.

     • Pre-LN layers (norm_first=True) for stable training without warmup.
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────
# Diffusion Schedule
# ─────────────────────────────────────────────────────────────────

class DiffusionSchedule:
    """
    Linear noise schedule:  β_s = β_start + (β_end − β_start) · s/S

    Precomputes ᾱ_s = ∏_{t=1}^{s} (1 − β_t) for fast q-sample.
    """

    def __init__(
        self,
        num_steps   : int    = 500,
        beta_start  : float  = 1e-4,
        beta_end    : float  = 0.02,
        sigma_pad   : float  = 0.01,   # noise std for starting/ending frames
        step_sample : str    = "sqrt_low",  # "sqrt_low" | "uniform" | "sqrt_high"
        device      : str    = "cpu",
    ):
        self.num_steps = num_steps
        self.sigma_pad = sigma_pad
        self.step_sample = step_sample

        betas     = torch.linspace(beta_start, beta_end, num_steps)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)          # (S,)

        self._sqrt_ab     = alpha_bar.sqrt()              # √ᾱ_s
        self._sqrt_1m_ab  = (1.0 - alpha_bar).sqrt()     # √(1−ᾱ_s)

        self.to(device)

    def to(self, device):
        self._sqrt_ab    = self._sqrt_ab.to(device)
        self._sqrt_1m_ab = self._sqrt_1m_ab.to(device)
        self._device     = device
        return self

    # ------------------------------------------------------------------
    def q_sample(
        self,
        y0         : torch.Tensor,   # (B, L, C) clean one-hot matrix
        step       : torch.Tensor,   # (B,)      1-indexed step
        valid_mask : torch.Tensor,   # (B, L)    True = valid frame
    ) -> torch.Tensor:
        """
        Corrupt y0 at the given diffusion step.

        Valid frames receive the standard forward-diffusion noise:
            Y_s = √ᾱ_s · Y_0 + √(1−ᾱ_s) · ε,   ε ~ N(0, I)

        Padding frames receive only tiny additive noise (σ_pad):
            Y_s_pad = Y_0 + σ_pad · ε_pad

        This keeps padding values close to their clean one-hot state so
        the model can reliably identify and ignore them.
        """
        idx = (step - 1).long()                               # 0-indexed

        sqrt_ab   = self._sqrt_ab[idx][:, None, None]         # (B,1,1)
        sqrt_1mab = self._sqrt_1m_ab[idx][:, None, None]      # (B,1,1)

        eps      = torch.randn_like(y0)
        y_valid  = sqrt_ab * y0 + sqrt_1mab * eps             # full noise

        eps_pad  = torch.randn_like(y0) * self.sigma_pad
        y_pad    = y0 + eps_pad                               # tiny noise

        # Select valid vs padding per frame
        vm = valid_mask.unsqueeze(-1).float()                 # (B,L,1)
        return vm * y_valid + (1.0 - vm) * y_pad

    # ------------------------------------------------------------------
    def sample_steps(self, batch_size: int, device: str) -> torch.Tensor:
        """
        Sample diffusion steps for training.

        step_sample controls the distribution over {1, ..., S}:
          "sqrt_low"  : p(s) ∝ 1/√s  — more low (less noisy) steps (default)
          "uniform"   : p(s) = 1/S   — equal exposure to all noise levels
          "sqrt_high" : p(s) ∝ √s    — more high (noisier) steps, better for
                        learning to denoise from pure noise
        """
        S = self.num_steps
        u = torch.rand(batch_size, device=device)

        if self.step_sample == "sqrt_low":
            # p(s) ∝ 1/√s  →  s = ceil(S·u²)
            steps = (S * u * u).long().clamp(0, S - 1) + 1
        elif self.step_sample == "uniform":
            # s ~ Uniform(1, S)
            steps = (u * S).long().clamp(0, S - 1) + 1
        elif self.step_sample == "sqrt_high":
            # p(s) ∝ √s  →  inverse CDF: s = (1 + u*(S^1.5 - 1))^(2/3)
            s_float = (1.0 + u * (S ** 1.5 - 1.0)) ** (2.0 / 3.0)
            steps = s_float.round().long().clamp(1, S)
        else:
            raise ValueError(
                f"step_sample must be 'sqrt_low', 'uniform', or 'sqrt_high'; got {self.step_sample!r}"
            )
        return steps


# ─────────────────────────────────────────────────────────────────
# Step Embedding
# ─────────────────────────────────────────────────────────────────

class StepEmbedding(nn.Module):
    """
    Maps a scalar diffusion step s ∈ {1,…,S} to a d_model vector.

    Pipeline:
        s  →  sinusoidal positional encoding  →  2-layer MLP  →  d_model
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s : (B,) integer step indices
        Returns (B, d_model)
        """
        half  = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=s.device, dtype=torch.float32) / half
        )                                                      # (half,)
        args = s[:, None].float() * freqs[None, :]            # (B, half)
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)
        return self.mlp(emb)                                   # (B, d_model)


# ─────────────────────────────────────────────────────────────────
# Denoising Transformer
# ─────────────────────────────────────────────────────────────────

class DiffusionTransformer(nn.Module):
    """
    Transformer-based denoising network.

    Architecture
    ────────────
        [Y_s ‖ time_feat] (B,L,19)  ──► Linear(19→d_model)
                                    ──► + learnable positional embedding
                                    ──► + step embedding (broadcast over L)
                                    ──► N × TransformerEncoderLayer
                                             └── src_key_padding_mask blocks starting/ending
                                    ──► Linear(d_model→18)
                                    ──► logits (B, L, 18)

    The extra 1 input feature is normalised time_hours (0→1 per patient).
    Padding frames have time_feat=0 and are masked out in attention.
    The caller applies softmax / log_softmax to get probabilities.
    """

    def __init__(
        self,
        num_classes    : int = 18,
        num_input_features: int = 19,   # 18 noisy classes + 1 time feature
        d_model        : int = 256,
        nhead          : int = 8,
        num_layers     : int = 6,
        dim_feedforward: int = 1024,
        dropout        : float = 0.1,
        max_len        : int = 550,
    ):
        super().__init__()

        # Input: concatenation of noisy stage vector + time feature
        # LayerNorm stabilises the wildly varying input scale across diffusion
        # steps: at step 1 values are near {0,1}, at step 100 they are ~N(0,1).
        # Without normalisation the input_proj sees very different distributions
        # depending on which step was sampled, making it hard to learn.
        self.input_norm = nn.LayerNorm(num_input_features)
        self.input_proj = nn.Linear(num_input_features, d_model)

        # Learnable positional embedding — one vector per frame position
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        self.step_emb = StepEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_feedforward,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # Pre-LN: more stable, no warmup needed
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, num_classes)

        # ── Logit mask ───────────────────────────────────────────────────────
        # Classes that should NEVER be predicted by the model:
        #   • starting / ending  — padding classes, not real embryo stages
        #   • tHB                — only 3 frames in entire dataset, not learnable
        #
        # Setting their logits to -1e9 before softmax/loss means:
        #   - They can never be argmax → never predicted
        #   - Their softmax probability ≈ 0 → don't pollute boundary dot-products
        #   - CE loss ignores them naturally (prob≈0 → loss≈0 for those positions)
        #
        # ALL_STAGE_COLS order:
        #   0:starting, 1:tPB2 … 16:tHB, 17:ending
        #
        # Blocked indices: 0 (starting), 16 (tHB), 17 (ending)
        logit_mask = torch.zeros(num_classes)   # 0 = allowed
        for blocked in [0, 16, 17]:             # starting, tHB, ending
            logit_mask[blocked] = 1.0
        # Register as buffer so it moves with .to(device) automatically
        self.register_buffer("logit_mask", logit_mask)

        # ── Break symmetry to avoid "always predict tPB2" collapse ───────────
        # When transformer output is near zero, logits are equal for allowed
        # classes; argmax then picks the first allowed index (1 = tPB2). Init
        # output bias with small random values for allowed classes (1..15) so
        # initial predictions are spread across stages and gradients can flow.
        with torch.no_grad():
            self.output_proj.bias.zero_()
            self.output_proj.bias[1:16].uniform_(-0.5, 0.5)
            self.output_proj.weight.mul_(0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        y_s         : torch.Tensor,   # (B, L, 18)  noisy stage matrix
        time_feat   : torch.Tensor,   # (B, L, 1)   normalised time_hours
        step        : torch.Tensor,   # (B,)         step indices
        padding_mask: torch.Tensor,   # (B, L)       True = starting/ending (ignored)
    ) -> torch.Tensor:
        """
        Returns logits (B, L, 18).

        time_feat is concatenated with y_s before projection so the model
        has patient-specific temporal context at every frame.

        padding_mask = True  means "this position is padding; ignore it as a key
        during attention".  Valid frames therefore never attend to padding frames
        and are not influenced by them.
        """
        B, L, _ = y_s.shape

        # Concatenate noisy stages with time feature → (B, L, 19)
        # then normalise before projection so scale is consistent across steps
        x = torch.cat([y_s, time_feat], dim=-1)
        x = self.input_norm(x)
        x = self.input_proj(x)                           # (B, L, d_model)
        x = x + self.pos_emb[:, :L, :]                  # add positional info
        x = x + self.step_emb(step)[:, None, :]         # broadcast step emb

        # src_key_padding_mask: True → that key position is masked out
        x = self.transformer(x, src_key_padding_mask=padding_mask)   # (B, L, d_model)

        logits = self.output_proj(x)                     # (B, L, 18)

        # Block invalid classes: set their logits to -1e9 so they can never
        # be predicted and don't pollute softmax probabilities
        logits = logits - self.logit_mask * 1e9           # (B, L, 18)

        return logits