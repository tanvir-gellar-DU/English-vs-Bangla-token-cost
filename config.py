"""One fixed configuration used for BOTH languages."""
MODEL = "qwen2.5:3b"
# Language-neutral system prompt; no "answer in the same language" instruction for either language.
SYSTEM = "You are a helpful assistant. Answer concisely."
OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 256}
