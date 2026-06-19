"""OpenFold embedder for extracting Evoformer pair representations."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from interplm.embedders.base import BaseEmbedder
from interplm.utils import get_device


class OpenFoldEmbedder(BaseEmbedder):
    """
    Embedder for OpenFold Evoformer pair representations.

    The primary output is the pair representation tensor with shape
    ``[batch, N, N, d_hidden]``, which is the input layout expected by
    ``SpatialPairSAE``.

    OpenFold installations and checkpoints vary by cluster, so this wrapper is
    deliberately thin:
    - If ``model`` is provided, it is used directly.
    - Otherwise, ``load_model`` attempts to instantiate OpenFold's AlphaFold
      module from ``openfold.config.model_config`` and
      ``openfold.model.model.AlphaFold``.
    - ``embed`` accepts either a full OpenFold feature dictionary or an integer
      MSA tensor shaped ``[B, M, N]``.
    """

    PAIR_OUTPUT_KEYS = (
        "pair",
        "pair_repr",
        "pair_representation",
        "pair_activations",
        "z",
    )

    def __init__(
        self,
        model_name: str = "model_1",
        device: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        pair_dim: Optional[int] = None,
        max_length: int = 1024,
        load_model: bool = True,
    ):
        if device is None:
            device = get_device()

        super().__init__(model_name=model_name, device=device)
        self.model = model
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.pair_dim = pair_dim
        self.max_length = max_length

        if load_model:
            self.load_model()

    def load_model(self) -> None:
        """Load an OpenFold AlphaFold model, optionally from a checkpoint."""
        if self.model is None:
            try:
                from openfold.config import model_config
                from openfold.model.model import AlphaFold
            except ImportError as exc:
                raise ImportError(
                    "OpenFold is not installed. Clone/install "
                    "https://github.com/aqlaboratory/openfold, or pass an "
                    "initialized OpenFold model via OpenFoldEmbedder(model=...)."
                ) from exc

            config = model_config(self.model_name, train=False)
            self.model = AlphaFold(config)

        if self.checkpoint_path is not None:
            state = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = state.get("state_dict", state)
            self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()

    def embed(self, msa_input: Union[torch.Tensor, Dict[str, Any]]) -> torch.Tensor:
        """
        Extract pair representations from an MSA or OpenFold feature batch.

        Args:
            msa_input: Either an integer MSA tensor shaped ``[B, M, N]`` or a
                full OpenFold feature dictionary. A feature dictionary is the
                most reliable path for production OpenFold runs because it can
                include templates, residue indices, masks, and recycling state.

        Returns:
            Pair representation shaped ``[B, N, N, d_hidden]``.
        """
        if self.model is None:
            raise RuntimeError("OpenFold model is not loaded")

        batch = (
            self._build_minimal_batch(msa_input)
            if torch.is_tensor(msa_input)
            else self._move_batch_to_device(msa_input)
        )

        with torch.no_grad():
            output = self.model(batch)

        pair = self._extract_pair_representation(output)
        pair = self._ensure_pair_layout(pair)
        if self.pair_dim is None:
            self.pair_dim = int(pair.shape[-1])
        return pair

    def _build_minimal_batch(self, msa_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Build a minimal OpenFold-like feature batch from integer MSA tokens.

        For production extraction, prefer passing the full feature dictionary
        produced by OpenFold's data pipeline. This helper exists for quick shape
        smoke tests and simple MSA-only inference paths.
        """
        if msa_tokens.ndim != 3:
            raise ValueError(
                f"Expected MSA tensor shape [batch, msa_depth, N], got {tuple(msa_tokens.shape)}"
            )

        msa_tokens = msa_tokens.to(self.device, dtype=torch.long)
        batch_size, msa_depth, seq_len = msa_tokens.shape

        residue_index = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len)
        seq_length = torch.full(
            (batch_size,),
            seq_len,
            dtype=torch.long,
            device=self.device,
        )
        msa_mask = torch.ones(batch_size, msa_depth, seq_len, device=self.device)
        seq_mask = torch.ones(batch_size, seq_len, device=self.device)

        # OpenFold's exact feature dimensions depend on the config. We provide
        # conservative one-hot token features; full pipeline feature dicts are
        # preferred when using a real OpenFold checkpoint.
        target_feat = F.one_hot(msa_tokens[:, 0, :].clamp(0, 20), num_classes=21).float()
        msa_feat = F.one_hot(msa_tokens.clamp(0, 20), num_classes=21).float()

        return {
            "aatype": msa_tokens[:, 0, :],
            "residue_index": residue_index,
            "seq_length": seq_length,
            "seq_mask": seq_mask,
            "msa": msa_tokens,
            "msa_mask": msa_mask,
            "target_feat": target_feat,
            "msa_feat": msa_feat,
        }

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _extract_pair_representation(self, output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in self.PAIR_OUTPUT_KEYS:
                if key in output and torch.is_tensor(output[key]):
                    return output[key]

            if "representations" in output and isinstance(output["representations"], dict):
                for key in self.PAIR_OUTPUT_KEYS:
                    value = output["representations"].get(key)
                    if torch.is_tensor(value):
                        return value

        if hasattr(output, "pair") and torch.is_tensor(output.pair):
            return output.pair

        raise KeyError(
            "Could not find a pair representation in OpenFold output. Expected "
            f"one of {self.PAIR_OUTPUT_KEYS}."
        )

    def _ensure_pair_layout(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim == 3:
            pair = pair.unsqueeze(0)
        if pair.ndim != 4:
            raise ValueError(
                f"Expected pair representation with 4 dims, got {tuple(pair.shape)}"
            )

        # Already [B, N, N, C].
        if pair.shape[1] == pair.shape[2]:
            return pair.contiguous()

        # Convert common channel-first layout [B, C, N, N] -> [B, N, N, C].
        if pair.shape[2] == pair.shape[3]:
            return pair.permute(0, 2, 3, 1).contiguous()

        raise ValueError(
            "Could not infer pair representation layout. Expected [B, N, N, C] "
            f"or [B, C, N, N], got {tuple(pair.shape)}"
        )

    def extract_embeddings(
        self,
        sequences: List[str],
        layer: int = -1,
        batch_size: int = 1,
        return_contacts: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError(
            "OpenFold pair extraction requires MSA/features. Use embed(msa_or_feature_batch)."
        )

    def embed_single_sequence(self, sequence: str, layer: int = -1) -> np.ndarray:
        raise NotImplementedError(
            "OpenFoldEmbedder requires an MSA or OpenFold feature batch, not a single raw sequence."
        )

    def embed_fasta_file(
        self,
        fasta_path: Path,
        layer: int = -1,
        output_path: Optional[Path] = None,
        batch_size: int = 1,
    ) -> Union[np.ndarray, None]:
        raise NotImplementedError(
            "OpenFoldEmbedder does not build MSAs from FASTA files. Generate OpenFold "
            "feature batches first, then call embed(...)."
        )

    def get_embedding_dim(self, layer: int = -1) -> int:
        if self.pair_dim is None:
            raise ValueError(
                "pair_dim is unknown until a pair representation is extracted; "
                "pass pair_dim=... if you need this before inference."
            )
        return self.pair_dim

    @property
    def available_layers(self) -> List[int]:
        return [-1]

    @property
    def max_sequence_length(self) -> int:
        return self.max_length

    def tokenize(self, sequences: List[str]) -> Dict:
        raise NotImplementedError(
            "OpenFold tokenization requires MSA/feature generation outside this embedder."
        )
