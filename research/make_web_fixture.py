"""Write the reference outputs the browser runtime is tested against (web/tests/parity.test.mjs).

Run after every Aura-Sense export: python make_web_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from neurocare.sense import AuraSense  # noqa: E402

TEXTS = [
    "I feel so alone tonight.", "My exam went GREAT and I'm thrilled!!!", "Café résumé naïve — über-tired, can't sleep...",
    "nothing ever works out for me, I always fail at everything I try", "I'm worried my friends secretly hate me",
    "日本語 mixed with English 123", "honestly? I'm fine. it's fine. everything's fine :)", "",
]


def main():
    sense = AuraSense()
    out = []
    for t in TEXTS:
        r = sense.read([t])[0]
        out.append({"text": t, "ids": sense.tokenizer.encode(t).ids, "risk": r.affect.risk, "valence": r.affect.valence,
                    "arousal": r.affect.arousal, "embedding8": r.embedding[:8].tolist()})
    path = ROOT.parent / "web" / "tests" / "fixtures" / "sense_fixture.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
