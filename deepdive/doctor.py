"""python -m deepdive.doctor  -> verifies config and that every model in every chain works through Mesh."""
from . import config, llm

# Report configuration problems first (missing key, empty chains).
for problem in config.validate():
    print("CONFIG", problem)

# Ping each model directly (bypassing fallback) so a broken id is obvious.
for role, models in config.CHAINS.items():
    for model in models:
        try:
            text, pt, ct, _ = llm._raw_call(model, "Reply with the single word: ok", "ping", False)
            print(f"OK    {role:10s} {model:32s} -> {text.strip()[:20]!r} (${llm.cost_usd(model, pt, ct):.6f})")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {role:10s} {model:32s} -> {str(e)[:100]}")
