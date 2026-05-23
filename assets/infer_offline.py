import argparse
from pathlib import Path

from mini_omni3.generate.base import run_inference


# Layout under <repo>/checkpoint/ — see README for what to put there.
_CKPT = Path(__file__).resolve().parent.parent / "checkpoint"
MODEL_CONFIG_DIR   = str(_CKPT / "model_config")
TRAINED_CHECKPOINT = str(_CKPT / "state_dict.pt")
QWEN_OMNI_CKPT     = str(_CKPT / "qwen2.5-omni_config")
AUDIO_TOWER_CKPT   = str(_CKPT / "audio_tower.pth")

# Edit this to the audio file you want to feed the model.
AUDIO_PATH = ""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="One-shot offline inference.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    args = p.parse_args()

    run_inference(
        model_config_dir=MODEL_CONFIG_DIR,
        trained_checkpoint=TRAINED_CHECKPOINT,
        qwen_omni_ckpt=QWEN_OMNI_CKPT,
        audio_tower_ckpt=AUDIO_TOWER_CKPT,
        audio_paths=[AUDIO_PATH],
        rounds=1,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )
