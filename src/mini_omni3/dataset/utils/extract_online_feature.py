"""Encode raw audio samples with Qwen2.5-Omni's audio tower; save to AudioFeat.pt.

Pure library — paths come in as arguments. The audio encoder is loaded once
per (qwen_omni_ckpt, audio_tower_ckpt, device) tuple and cached for reuse.
"""

import os

import numpy as np
import torch
import whisper
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration


_encoder_cache = {}  # keyed by (qwen_omni_ckpt, audio_tower_ckpt, str(device))


def _load_encoder(qwen_omni_ckpt, audio_tower_ckpt, device):
    cfg = AutoConfig.from_pretrained(qwen_omni_ckpt)
    enc = Qwen2_5OmniForConditionalGeneration._from_config(cfg).thinker.audio_tower
    enc.load_state_dict(torch.load(audio_tower_ckpt, map_location=device))
    return enc.to(device).requires_grad_(False).eval()


def _split_into_chunks(n, chunk_size):
    chunks = [chunk_size] * (n // chunk_size)
    if n % chunk_size:
        chunks.append(n % chunk_size)
    return chunks


def extract_audio_features(audio_samples, save_dir, *,
                           qwen_omni_ckpt, audio_tower_ckpt, device="cuda"):
    """Encode raw audio samples; save the feature tensor to `<save_dir>/AudioFeat.pt`."""
    key = (qwen_omni_ckpt, audio_tower_ckpt, str(device))
    if key not in _encoder_cache:
        _encoder_cache[key] = _load_encoder(qwen_omni_ckpt, audio_tower_ckpt, device)
    encoder = _encoder_cache[key]

    mel = whisper.log_mel_spectrogram(np.array(audio_samples, dtype=np.float32), n_mels=128)
    len_feature = mel.shape[1]

    with torch.no_grad():
        feat = encoder(
            torch.tensor(mel).to(device),
            torch.tensor(_split_into_chunks(len_feature, 40)).to(device),
            torch.tensor((len_feature - 1) // 2 + 1).to(device),
        ).last_hidden_state

    os.makedirs(save_dir, exist_ok=True)
    torch.save(feat.detach().cpu(), os.path.join(save_dir, "AudioFeat.pt"))
