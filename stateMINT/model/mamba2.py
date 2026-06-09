import flax.nnx as nnx
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from mamba2_jax import Mamba2Config, Mamba2Model


class Mamba2Regressor(nnx.Module):
    """Per-timestep sequence regressor built on the Mamba2 backbone.

    Input:  (batch, T, input_dim)
    Output: (batch, T, output_dim)

    Mamba2 is causal, so the prediction at timestep ``t`` only depends on
    timesteps ``<= t`` — the right inductive bias for forecasting prevalence
    / cases as the simulation evolves. Targets are predicted in the
    transformed space (logit for prevalence, log1p for cases); apply the
    inverse transform at eval time.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        head_dim: int = 64,
        chunk_size: int = 256,
        output_dim: int = 1,
        dropout: float = 0.05,
        *,
        rngs: nnx.Rngs,
    ):
        assert (d_model * expand) % head_dim == 0, "d_model * expand must be divisible by head_dim"

        self.input_proj = nnx.Linear(input_dim, d_model, rngs=rngs)
        cfg = Mamba2Config(
            vocab_size=1,  # unused: we feed inputs_embeds, not token ids
            hidden_size=d_model,
            state_size=d_state,
            head_dim=head_dim,
            expand=expand,
            conv_kernel=d_conv,
            chunk_size=chunk_size,
            num_hidden_layers=n_layers,
        )
        self.mamba2 = Mamba2Model(cfg, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.output_proj = nnx.Linear(d_model, output_dim, rngs=rngs)  # TODO: maybe more layers here?

    # TODO: maybe make bidirectional?
    @jax.named_scope("Mamba2Regressor")
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = self.input_proj(x)  # (B, T, d_model)
        h = self.mamba2(input_ids=None, inputs_embed=h)["last_hidden_state"]  # (B, T, d_model) - full sequence
        h = self.dropout(h)
        return self.output_proj(h)  # (B, T, output_dim)

    @classmethod
    def from_cfg(cls, cfg: DictConfig, input_size: int) -> "Mamba2Regressor":
        return cls(
            input_dim=input_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            head_dim=cfg.head_dim,
            chunk_size=cfg.chunk_size,
            output_dim=cfg.output_dim,
            dropout=cfg.dropout,
            rngs=nnx.Rngs(cfg.seed),
        )
