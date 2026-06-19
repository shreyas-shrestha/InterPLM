"""
Trainer for convolutional SAEs over OpenFold pair representations.

SpatialPairTrainer expects pair activations with shape [batch, N, N, d_hidden]
and optimizes the SpatialPairSAE with an MSE reconstruction objective plus an
L1 sparsity penalty on the sparse convolutional feature maps.
"""

from collections import namedtuple
from dataclasses import dataclass

import torch as t
import torch.nn.functional as F

from interplm.sae.dictionary import SpatialPairSAE
from interplm.train.trainers.base_trainer import SAETrainer, SAETrainerConfig
from interplm.train.trainers.common import get_lr_schedule, get_sparsity_warmup_fn
from interplm.utils import get_device


@dataclass
class SpatialPairTrainerConfig(SAETrainerConfig):
    """
    Configuration for SpatialPairTrainer.

    The base SAETrainerConfig fields are reused where possible:
    - activation_dim is the OpenFold pair channel dimension d_hidden.
    - expansion_factor controls d_dictionary = d_hidden * expansion_factor.
    """

    l1_penalty: float = 0.06
    l1_penalty_warmup_steps: int | None = None
    trainer_name: str = "SpatialPairTrainer"

    def set_and_validate_activation_dim(self, activation_dim: int):
        super().set_and_validate_activation_dim(activation_dim)

        if int(self.expansion_factor) != self.expansion_factor:
            raise ValueError(
                "SpatialPairSAE requires dictionary_size to be an integer multiple "
                "of activation_dim so expansion_factor is an integer"
            )
        self.expansion_factor = int(self.expansion_factor)

    @classmethod
    def trainer_cls(cls) -> type["SpatialPairTrainer"]:
        return SpatialPairTrainer


class SpatialPairTrainer(SAETrainer):
    """
    Trainer for SpatialPairSAE on 2D pair representations.

    Loss:
        total = MSE(x, x_hat) + l1_penalty * mean(abs(f_x))

    Metrics:
        - MSE reconstruction loss
        - L1 sparsity loss
        - L0 sparsity averaged per pair position
        - Variance explained by the reconstruction
    """

    def __init__(self, trainer_config: SpatialPairTrainerConfig):
        super().__init__(
            trainer_config=trainer_config,
            logging_parameters=["training/learning_rate", "training/l1_penalty"],
        )

        self.lr = trainer_config.lr
        self.steps = trainer_config.steps
        self.warmup_steps = trainer_config.warmup_steps
        self.decay_start = trainer_config.decay_start
        self.grad_clip_norm = trainer_config.grad_clip_norm

        self.ae = SpatialPairSAE(
            d_hidden=trainer_config.activation_dim,
            expansion_factor=trainer_config.expansion_factor,
            normalize_to_sqrt_d=trainer_config.normalize_to_sqrt_d,
        )

        self.device = get_device()
        self.ae.to(self.device)

        if trainer_config.l1_penalty_warmup_steps is None:
            trainer_config.l1_penalty_warmup_steps = int(self.steps * 0.05)

        self.l1_penalty = trainer_config.l1_penalty
        self.l1_penalty_warmup_steps = trainer_config.l1_penalty_warmup_steps
        self.l1_penalty_warmup_fn = get_sparsity_warmup_fn(
            self.steps, self.l1_penalty_warmup_steps
        )
        self.current_l1_penalty_scale = 0.0

        self.steps_since_active = t.zeros(
            self.ae.dict_size, dtype=t.long, device=self.device
        )
        self.l0_sparsity = 0.0
        self.mse = 0.0
        self.variance_explained = 0.0

        self.optimizer = t.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        lr_fn = get_lr_schedule(
            total_steps=self.steps,
            warmup_steps=self.warmup_steps,
            decay_start=self.decay_start,
        )
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        print(f"Training with config: {self.config}")

    @classmethod
    def dictionary_cls(cls):
        return SpatialPairSAE

    def _validate_pair_batch(self, x: t.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"SpatialPairTrainer expected [batch, N, N, d_hidden], got {tuple(x.shape)}"
            )
        if x.shape[1] != x.shape[2]:
            raise ValueError(
                f"SpatialPairTrainer expected a square pair grid, got {x.shape[1]} x {x.shape[2]}"
            )
        if x.shape[-1] != self.ae.activation_dim:
            raise ValueError(
                f"SpatialPairTrainer expected d_hidden={self.ae.activation_dim}, got {x.shape[-1]}"
            )

    def _calculate_l0_sparsity(self, f: t.Tensor) -> t.Tensor:
        """
        Average number of active dictionary features per residue-pair position.

        f has shape [B, d_dictionary, N, N]. Summing over dim=1 counts how many
        sparse features are non-zero at each [batch, i, j] pair position.
        """
        return (f != 0).float().sum(dim=1).mean()

    def _calculate_variance_explained(self, x: t.Tensor, x_hat: t.Tensor) -> t.Tensor:
        x_float = x.float()
        residual = (x - x_hat).float()
        total_variance = t.var(x_float)
        residual_variance = t.var(residual)

        if total_variance <= 0:
            return t.tensor(0.0, dtype=x_float.dtype, device=x.device)
        return 1 - residual_variance / total_variance

    def loss(self, x: t.Tensor, step: int | None = None, logging: bool = False):
        """
        Compute the SpatialPairSAE training loss for one pair-representation batch.

        Args:
            x: OpenFold pair representations with shape [B, N, N, d_hidden].
            step: Current training step, used for optional L1 warmup.
            logging: If True, return the reconstructed tensor, activations, and
                scalar metric dictionary in addition to the input batch.
        """
        self._validate_pair_batch(x)
        if step is None:
            step = 0

        x_hat, f = self.ae(x)

        reconstruction_loss = F.mse_loss(x_hat.float(), x.float())
        l1_loss = f.abs().mean()
        l0_sparsity = self._calculate_l0_sparsity(f)
        variance_explained = self._calculate_variance_explained(x, x_hat)

        did_fire = (f != 0).any(dim=(0, 2, 3))
        self.steps_since_active += 1
        self.steps_since_active[did_fire] = 0

        self.current_l1_penalty_scale = self.l1_penalty * self.l1_penalty_warmup_fn(
            step
        )
        total_loss = reconstruction_loss + self.current_l1_penalty_scale * l1_loss

        self.l0_sparsity = l0_sparsity.item()
        self.mse = reconstruction_loss.item()
        self.variance_explained = variance_explained.item()

        if not logging:
            return total_loss

        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
            x,
            x_hat,
            f,
            {
                "loss/reconstruction": reconstruction_loss.item(),
                "loss/sparsity": l1_loss.item(),
                "loss/total": total_loss.item(),
                "performance/mse": reconstruction_loss.item(),
                "performance/l0_sparsity": l0_sparsity.item(),
                "performance/variance_explained": variance_explained.item(),
            },
        )

    def update(self, step: int, x: t.Tensor) -> float:
        x = x.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(x, step=step)
        loss.backward()

        if self.grad_clip_norm is not None:
            t.nn.utils.clip_grad_norm_(self.ae.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def get_per_dimension_mse(self, x: t.Tensor) -> t.Tensor:
        """
        Calculate final per-channel MSE for pair representations.

        Returns:
            Tensor with shape [d_hidden], averaged over batch and N x N positions.
        """
        x = x.to(self.device)
        self._validate_pair_batch(x)

        with t.no_grad():
            x_hat, _ = self.ae(x)
            return (x - x_hat).pow(2).mean(dim=(0, 1, 2))
