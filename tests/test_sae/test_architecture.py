"""
Tests for SAE dictionary architecture.

Tests only the dictionary/SAE classes themselves - initialization,
architectural constraints, and architecture-specific behavior.

Trainer-specific tests belong in test_trainers.py.
"""

import pytest
import torch
import numpy as np

from interplm.sae.dictionary import (
    ReLUSAE,
    ReLUSAE_Tied,
    SpatialPairSAE,
    TopKSAE,
    BatchTopKSAE,
    JumpReLUSAE,
)


# ============================================================================
# INITIALIZATION
# ============================================================================

@pytest.mark.parametrize("sae_class,sae_kwargs", [
    (ReLUSAE, {'activation_dim': 32, 'dict_size': 64}),
    (TopKSAE, {'activation_dim': 32, 'dict_size': 64, 'k': 8}),
    (BatchTopKSAE, {'activation_dim': 32, 'dict_size': 64, 'k': 8}),
    (JumpReLUSAE, {'activation_dim': 32, 'dict_size': 64}),
])
def test_decoder_initialized_to_unit_norm(sae_class, sae_kwargs):
    """Test that all SAE decoder weights are unit-normalized at initialization.

    All SAE types normalize so that each FEATURE has unit norm.
    With nn.Linear storing weights as (activation_dim, dict_size), this means
    normalizing along dim=0 (columns), giving dict_size unit norms.
    """
    sae = sae_class(**sae_kwargs)

    # All SAE types: norm along columns (dim=0), one norm per feature
    norm_dim = 0
    expected_size = sae_kwargs['dict_size']

    decoder_norms = torch.norm(sae.decoder.weight.data, dim=norm_dim)
    expected_norms = torch.ones(expected_size)

    assert torch.allclose(decoder_norms, expected_norms, atol=1e-6), \
        f"{sae_class.__name__}: Decoder weights should be unit-normalized at init (one norm per feature)"


@pytest.mark.parametrize("sae_class,sae_kwargs", [
    (TopKSAE, {'activation_dim': 32, 'dict_size': 64, 'k': 8}),
    (BatchTopKSAE, {'activation_dim': 32, 'dict_size': 64, 'k': 8}),
    (JumpReLUSAE, {'activation_dim': 32, 'dict_size': 64}),
])
def test_encoder_initialized_as_decoder_transpose(sae_class, sae_kwargs):
    """Test that encoder = decoder.T at initialization for tied-weight SAEs.

    Note: ReLUSAE has untied weights, so encoder is NOT initialized as decoder.T.
    """
    sae = sae_class(**sae_kwargs)

    encoder_weights = sae.encoder.weight.data
    decoder_weights = sae.decoder.weight.data.T

    assert torch.allclose(encoder_weights, decoder_weights, atol=1e-6), \
        f"{sae_class.__name__}: Encoder should equal decoder.T at init"


def test_relu_tied_weights_tied_at_initialization():
    """Test that ReLUSAE_Tied has encoder.weight = decoder.weight.T at init."""
    sae = ReLUSAE_Tied(activation_dim=32, dict_size=64)

    assert torch.allclose(sae.encoder.weight.data, sae.decoder.weight.data.T, atol=1e-6), \
        "ReLUSAE_Tied: Encoder should equal decoder.T at init"


# ============================================================================
# JUMPRELU ARCHITECTURE-SPECIFIC BEHAVIOR
# ============================================================================

def test_jumprelu_threshold_initialization():
    """Test that JumpReLU threshold is initialized to 0.001."""
    sae = JumpReLUSAE(activation_dim=32, dict_size=64)

    # Threshold should be initialized to 0.001 for all features
    expected_threshold = torch.ones(64) * 0.001
    assert torch.allclose(sae.threshold, expected_threshold, atol=1e-6), \
        "JumpReLUSAE: threshold should be initialized to 0.001"


def test_jumprelu_threshold_is_learnable():
    """Test that JumpReLU threshold is a learnable parameter."""
    sae = JumpReLUSAE(activation_dim=32, dict_size=64)

    # Threshold should be a parameter
    assert isinstance(sae.threshold, torch.nn.Parameter), \
        "JumpReLUSAE: threshold should be a nn.Parameter"

    # Should have requires_grad=True
    assert sae.threshold.requires_grad, \
        "JumpReLUSAE: threshold should be learnable (requires_grad=True)"


# ============================================================================
# SPATIAL PAIR SAE ARCHITECTURE-SPECIFIC BEHAVIOR
# ============================================================================

def test_spatial_pair_sae_shapes_and_biases():
    """Test SpatialPairSAE preserves pair-grid shape and uses custom biases."""
    batch_size = 2
    seq_len = 7
    d_hidden = 8
    expansion_factor = 4

    sae = SpatialPairSAE(d_hidden=d_hidden, expansion_factor=expansion_factor)
    pair_repr = torch.randn(batch_size, seq_len, seq_len, d_hidden)

    reconstruction, features = sae(pair_repr)

    assert sae.dict_size == d_hidden * expansion_factor
    assert sae.activation_dim == d_hidden
    assert reconstruction.shape == pair_repr.shape
    assert features.shape == (
        batch_size,
        d_hidden * expansion_factor,
        seq_len,
        seq_len,
    )
    assert sae.encoder.bias is None
    assert sae.decoder.bias is None
    assert sae.encoder_bias.shape == (d_hidden * expansion_factor,)
    assert sae.decoder_bias.shape == (d_hidden,)


def test_spatial_pair_sae_forward_matches_manual_formula():
    """Test SpatialPairSAE implements ReLU(W_e(x - b_d) + b_e) and adds b_d back."""
    sae = SpatialPairSAE(d_hidden=4, expansion_factor=2)
    pair_repr = torch.randn(3, 5, 5, 4)

    reconstruction, features = sae(pair_repr)

    x_channels_first = pair_repr.permute(0, 3, 1, 2)
    shifted_x = x_channels_first - sae.decoder_bias.view(1, -1, 1, 1)
    expected_features = torch.relu(
        sae.encoder(shifted_x) + sae.encoder_bias.view(1, -1, 1, 1)
    )
    expected_reconstruction = (
        sae.decoder(expected_features) + sae.decoder_bias.view(1, -1, 1, 1)
    ).permute(0, 2, 3, 1)

    assert torch.allclose(features, expected_features, atol=1e-6)
    assert torch.allclose(reconstruction, expected_reconstruction, atol=1e-6)


# ============================================================================
# NORMALIZATION
# ============================================================================

@pytest.mark.parametrize("sae_class,sae_kwargs", [
    (ReLUSAE, {'activation_dim': 32, 'dict_size': 64}),
    (BatchTopKSAE, {'activation_dim': 32, 'dict_size': 64, 'k': 8}),
])
def test_normalize_to_sqrt_d_makes_reconstruction_scale_invariant(sae_class, sae_kwargs):
    """Test that normalize_to_sqrt_d makes reconstruction scale-invariant.

    The KEY property: inputs that differ only by scale should produce
    nearly identical normalized reconstructions (when unnormalize=False).

    This tests the core value of normalization - it removes scale as a confound.

    Note: Only tests SAE types that support unnormalize=False (ReLUSAE, BatchTopKSAE).
    TopKSAE and JumpReLUSAE always unnormalize outputs, so they don't support this mode.
    """
    # Create SAE with normalization enabled
    sae = sae_class(**sae_kwargs, normalize_to_sqrt_d=True)
    d = sae_kwargs['activation_dim']

    # Same base embeddings, different scales
    base_embeddings = torch.randn(10, d)
    scales = [0.1, 1.0, 10.0]  # 100x range

    normalized_reconstructions = []
    for scale in scales:
        scaled_input = base_embeddings * scale
        # Get normalized reconstruction (stays in sqrt(d)-normalized space)
        recon = sae.forward(scaled_input, unnormalize=False)
        normalized_reconstructions.append(recon)

    # All normalized reconstructions should be nearly identical
    baseline = normalized_reconstructions[1]  # 1.0x scale
    for i, scale in enumerate(scales):
        if i == 1:  # Skip baseline vs itself
            continue

        # Measure relative difference
        diff = torch.norm(normalized_reconstructions[i] - baseline, dim=-1).mean()
        baseline_norm = torch.norm(baseline, dim=-1).mean()
        relative_diff = diff / (baseline_norm + 1e-8)

        assert relative_diff < 0.2, \
            f"{sae_class.__name__}: Scale {scale}x should produce similar reconstruction. " \
            f"Relative diff: {relative_diff:.3f}"


@pytest.mark.parametrize("sae_class,sae_kwargs", [
    (ReLUSAE, {'activation_dim': 32, 'dict_size': 64}),
    (BatchTopKSAE, {'activation_dim': 32, 'dict_size': 64, 'k': 8}),
])
def test_unnormalize_restores_original_scale(sae_class, sae_kwargs):
    """Test that unnormalize=True restores outputs to original input scale.

    Critical: When unnormalize=True, the reconstruction norms should vary with
    input norms. When unnormalize=False, they should be normalized to sqrt(d).

    Note: Only tests SAE types that support unnormalize parameter (ReLUSAE, BatchTopKSAE).
    TopKSAE and JumpReLUSAE always unnormalize, so they don't support this mode.
    """
    sae = sae_class(**sae_kwargs, normalize_to_sqrt_d=True)
    d = sae_kwargs['activation_dim']

    # Create inputs with varying norms
    embeddings = torch.randn(20, d)
    scales = torch.linspace(0.5, 3.0, 20).unsqueeze(1)
    embeddings_varied = embeddings * scales

    input_norms = torch.norm(embeddings_varied, dim=-1)

    # Get reconstructions with and without unnormalize
    recon_unnormalized = sae.forward(embeddings_varied, unnormalize=True)
    recon_normalized = sae.forward(embeddings_varied, unnormalize=False)

    unnorm_norms = torch.norm(recon_unnormalized, dim=-1)
    norm_norms = torch.norm(recon_normalized, dim=-1)

    # Key test: Unnormalized should have more variable norms (following input)
    # Normalized should have more consistent norms (around sqrt(d))
    unnorm_std = unnorm_norms.std()
    norm_std = norm_norms.std()

    assert unnorm_std > norm_std * 1.5, \
        f"{sae_class.__name__}: Unnormalized norms should be more variable. " \
        f"Unnorm std: {unnorm_std:.2f}, Norm std: {norm_std:.2f}"

    # Input norms should be highly variable (we created them that way)
    assert input_norms.std() > 1.0, "Input norms should be variable"

    # Normalized output norms should be much more consistent than input
    assert norm_std < input_norms.std() / 2, \
        f"{sae_class.__name__}: Normalized outputs should have consistent norms"


# ============================================================================
# DICTIONARY INTERFACE
# ============================================================================

def test_encode_feat_subset_matches_full(small_relu_sae):
    """Test that encode_feat_subset gives same results as slicing full encoding."""
    sae = small_relu_sae
    embeddings = torch.randn(10, 32)

    # Encode all features
    full_features = sae.encode(embeddings)

    # Encode subset
    feat_list = [0, 5, 10, 15, 20]
    subset_features = sae.encode_feat_subset(embeddings, feat_list)

    # Should match
    expected = full_features[:, feat_list]
    assert torch.allclose(subset_features, expected, atol=1e-6), \
        f"Subset encoding doesn't match full encoding"


def test_normalize_features_flag(small_relu_sae):
    """Test that normalize_features=True divides by rescale factors."""
    sae = small_relu_sae
    embeddings = torch.randn(5, 32)

    # Set some non-trivial rescale factors
    sae.activation_rescale_factor = torch.tensor([2.0] * 64)

    # Encode without normalization
    features_unnormalized = sae.encode(embeddings, normalize_features=False)

    # Encode with normalization
    features_normalized = sae.encode(embeddings, normalize_features=True)

    # Normalized should be unnormalized divided by rescale factors
    expected = features_unnormalized / sae.activation_rescale_factor
    assert torch.allclose(features_normalized, expected, atol=1e-6), \
        "Normalized features don't match expected division"
