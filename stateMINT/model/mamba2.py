import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from mamba2_jax import Mamba2Config, Mamba2Model
from omegaconf import DictConfig

from stateMINT.common.dataclasses import Predictor
from stateMINT.model.hub import ModelArtifact, load_model_artifact


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
        """
        Initialize the Mamba2 regressor.

        Args:
            input_dim: Input feature size.
            d_model: Hidden size.
            n_layers: Number of Mamba2 layers.
            d_state: State size.
            d_conv: Convolution kernel size.
            expand: Expansion factor.
            head_dim: Head dimension.
            chunk_size: Mamba chunk size.
            output_dim: Output feature size.
            dropout: Dropout rate.
            rngs: Flax random streams.

        Returns:
            None.
        """
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
        self.output_proj = nnx.Linear(d_model, output_dim, rngs=rngs)

    @jax.named_scope("Mamba2Regressor")
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Predict one output per timestep.

        Args:
            x: Input batch with shape (B, T, input_dim).

        Returns:
            Predictions with shape (B, T, output_dim).
        """
        h = self.input_proj(x)  # (B, T, d_model)
        h = self.mamba2(input_ids=None, inputs_embeds=h)["last_hidden_state"]  # (B, T, d_model) - full sequence
        h = self.dropout(h)
        return self.output_proj(h)  # (B, T, output_dim)

    @classmethod
    def from_cfg(cls, cfg: DictConfig, input_size: int) -> "Mamba2Regressor":
        """
        Build a regressor from config.

        Args:
            cfg: Model config.
            input_size: Input feature size.

        Returns:
            Configured regressor.
        """
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
            rngs=nnx.Rngs(cfg.get("seed", 0)),
        )

    @classmethod
    def from_pretrained(
        cls: type["Mamba2Regressor"],
        path_or_repo_id: str,
        predictor: Predictor,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        local_dir: str | None = None,
    ) -> ModelArtifact:
        """
        Load a pretrained model from a local folder or Hugging Face repo.

        Example usage:
        ```
        Mamba2Regressor.from_pretrained("dide-ic/stateMINT", predictor="prevalence")
        Mamba2Regressor.from_pretrained("dide-ic/stateMINT", predictor="cases", revision="v1.0.0")
        Mamba2Regressor.from_pretrained("dide-ic/stateMINT", predictor="prevalence", local_dir="/path/to/local/dir")
        ```

        Args:
            path_or_repo_id: Hugging Face repo ID or local folder path.
            predictor: Target predictor, either "prevalence" or "cases".
            revision: Optional revision of the model to load from the repo.
            cache_dir: Optional cache directory for Hugging Face repo.
            local_dir: Optional local directory to load the artifact from.
            return_artifact: If True, return a ModelArtifact dataclass instead of just the model.

        Returns:
            ModelArtifact containing the loaded model and its config.
        """
        return load_model_artifact(
            path_or_repo_id,
            predictor,
            model_cls=cls,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
        )


def get_total_params(model: nnx.Module) -> int:
    """
    Get the total number of parameters in the model.

    Args:
        model: Flax module.

    Returns:
        Total parameter count.
    """
    params = nnx.state(model, nnx.Param)
    return sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
