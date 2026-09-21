import torch

from manual_branch.src.audio import adjust_duration
from manual_branch.src.log_mel import LogMelExtractor
from manual_branch.src.losses import supervised_contrastive_loss
from manual_branch.src.manual_encoder import ManualEncoder


def test_audio_and_log_mel_shapes():
    waveform = adjust_duration(torch.randn(32000))
    assert waveform.shape == (64000,)
    features = LogMelExtractor()(waveform)
    assert features.shape == (80, 401)
    assert torch.isfinite(features).all()


def test_encoder_contract():
    embedding = ManualEncoder()(torch.randn(4, 80, 401))
    assert embedding.shape == (4, 192)
    assert embedding.dtype == torch.float32
    assert torch.isfinite(embedding).all()
    torch.testing.assert_close(embedding.norm(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)


def test_independent_contrastive_loss():
    embedding = ManualEncoder()(torch.randn(4, 80, 401))
    losses = supervised_contrastive_loss(embedding, ["rasim", "rasim", "alice", "alice"])
    assert set(losses) == {"loss_total", "loss_cosine", "loss_mse"}
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()


def test_silence_is_finite():
    features = LogMelExtractor()(adjust_duration(torch.zeros(16000)))
    embedding = ManualEncoder()(features.unsqueeze(0))[0]
    assert torch.isfinite(embedding).all()
    torch.testing.assert_close(embedding.norm(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_audio_and_log_mel_shapes()
    test_encoder_contract()
    print("Pipeline vérifié")
