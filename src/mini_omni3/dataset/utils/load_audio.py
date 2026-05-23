"""
Shared audio loading helpers for the dataset preprocessing scripts.

The audio_tower in Qwen2.5-Omni does two stride-2 conv downsamples, so a mel
spectrogram of length L produces ((L-1)//2 + 1 - 2)//2 + 1 encoder-output
frames. We refer to these encoder-output frames just as "frames" throughout.
"""

import librosa
import numpy as np
import whisper


# Number of raw audio samples (@ 16 kHz) per encoder-output frame.
# 16000 Hz * 40 ms/frame = 640 samples / frame.
SAMPLES_PER_FRAME = 640


def _count_output_lengths(input_lengths: int):
    """Two-step downsample matching the audio_tower's conv stack."""
    input_lengths = (input_lengths - 1) // 2 + 1
    output_lengths = (input_lengths - 2) // 2 + 1
    return input_lengths, output_lengths


def _load_mel(audio_path: str, max_seconds: float = None):
    """Load audio @ 16 kHz, compute log-mel + downsampled lengths.

    Args:
        audio_path: path to audio file readable by librosa.
        max_seconds: optional truncation of the raw audio (offline samples
            cap at 20 s; online uses the full clip).

    Returns:
        audio        : list[float], raw audio samples (post-truncation, pre-mel-padding).
        mel          : torch.Tensor of shape (128, len_feature), log-mel spectrogram.
        len_feature  : int, mel.shape[1] (the mel time axis length).
        input_len    : int, length after the first conv downsample.
        output_len   : int, length after both conv downsamples (encoder frames).
    """
    audio_np, _ = librosa.load(audio_path, sr=16000)
    audio = audio_np.tolist()

    if max_seconds is not None:
        audio = audio[: int(max_seconds * 16000)]

    # Pad to a 160-sample multiple for whisper's mel-hop alignment.
    audio_for_mel = (
        audio if len(audio) % 160 == 0
        else audio + [0] * (160 - len(audio) % 160)
    )
    mel = whisper.log_mel_spectrogram(np.array(audio_for_mel, dtype=np.float32), n_mels=128)
    len_feature = mel.shape[1]
    input_len, output_len = _count_output_lengths(len_feature)

    return audio, mel, len_feature, input_len, output_len
