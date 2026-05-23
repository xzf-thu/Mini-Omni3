"""
Single source of truth for special token IDs used by the audio-enhanced GPT.

Both `cons_online_data.py` and `cons_offline_data.py` import from here so
that any change to a token id propagates everywhere.
"""

# === Vocabulary layout ===
VOCAB_SHIFT = 151600

# === Role / segment markers ===
USER = VOCAB_SHIFT + 1
ASSISTANT = VOCAB_SHIFT + 2

TEXT_BEGIN = 151644
TEXT_END = 151643
SYSTEM = TEXT_BEGIN  # 151644 — same id used as both turn separator and system marker

# === Audio markers ===
AUDIO_BEGIN = 151647
AUDIO_END = 151648
AUDIO_PAD = 151646

# === Control / padding ===
PAD = VOCAB_SHIFT + 8
MASK = -100
KEEP_SILENCE = VOCAB_SHIFT + 5

# === Task mode (offline batch vs online streaming) ===
ONLINE = VOCAB_SHIFT + 9
OFFLINE = VOCAB_SHIFT + 10

# === Language tag ===
ENGLISH = VOCAB_SHIFT + 11

# === Emotion tags (online only) ===
HAPPY = VOCAB_SHIFT + 14
SAD = VOCAB_SHIFT + 15
ANGRY = VOCAB_SHIFT + 16
SURPRISE = VOCAB_SHIFT + 17
NORMAL = VOCAB_SHIFT + 18
URGENT = VOCAB_SHIFT + 19

EMOTION_TO_ID = {
    "surprise": SURPRISE,
    "happy":    HAPPY,
    "normal":   NORMAL,
    "sad":      SAD,
    "angry":    ANGRY,
    "urgent":   URGENT,
}
