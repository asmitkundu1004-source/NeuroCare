"""AGEM benchmark: does memory recall help without resurfacing pain at the wrong time?

A scripted personal history (20 disclosures, half painful) is embedded with the
deployed Aura-Sense model and probed with four kinds of query:

  relevant     on the topic of one memory              -> it should be recalled
  unrelated    everyday small talk                     -> nothing painful should surface
  tangential   happy news that brushes a painful topic -> the painful memory should stay quiet
  follow-up    the same topic three turns in a row     -> no need to re-inject every turn

The history is author-written and small; it measures the behaviour of the ranking
policy rather than the embedding model, and is reported as such.

Metrics
  recall@3      share of relevant queries whose target memory is injected
  ISR           inappropriate surfacing rate: share of unrelated + tangential queries
                where any painful memory is injected
  empty_rate    share of unrelated queries where nothing is injected at all
  redundancy    share of follow-up turns that re-inject a memory injected the turn before

Run: python eval_memory.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from neurocare.agem import AgemParams, fixed_threshold, naive_top1, rank  # noqa: E402
from neurocare.sense import AuraSense  # noqa: E402

RESULTS = ROOT / "results"

MEMORIES = [
    # painful
    ("grandmother", "My grandmother passed away last month and I still can't believe she's gone"),
    ("breakup", "My girlfriend broke up with me in March and I still think about her every night"),
    ("driving", "I failed my driving test for the second time and felt so stupid"),
    ("parents", "My parents keep fighting and I hate being at home"),
    ("internship", "I got rejected from the internship I really wanted"),
    ("panic", "I had a panic attack during my presentation last week"),
    ("friend", "My best friend stopped talking to me and I don't know why"),
    ("sleep", "I've been struggling to sleep because of stress at work"),
    ("dog", "My dog was sick and we had to put him down"),
    ("lonely", "I feel lonely since I moved to Bangalore for my job"),
    # neutral or pleasant
    ("guitar", "I started learning the guitar and I'm loving it"),
    ("sister", "My sister is getting married in December"),
    ("running", "I've been going for morning runs by the lake"),
    ("offer", "I got a new job offer from a startup in Pune"),
    ("exam", "My data structures exam is on Friday"),
    ("kitten", "I adopted a kitten named Mishti"),
    ("cooking", "I've been cooking more at home and trying new recipes"),
    ("trip", "My friends and I are planning a trip to Darjeeling"),
    ("book", "I finished reading a book about stoicism"),
    ("swim", "I'm learning to swim this summer"),
]

RELEVANT = {
    "grandmother": ["I keep thinking about my grandma today", "Losing my grandmother still hurts so much"],
    "breakup": ["I miss my ex so much, I can't stop thinking about her", "The breakup is still on my mind every night"],
    "driving": ["I'm nervous about retaking my driving test", "Why do I keep failing the driving exam"],
    "parents": ["My mum and dad were shouting at each other again tonight", "Home feels so tense because of my parents"],
    "internship": ["I still feel bad about that internship rejection", "Should I apply to internships again after being rejected"],
    "panic": ["I'm scared I'll have another panic attack when I present", "My heart races before presentations now"],
    "friend": ["I wish I knew why my best friend went quiet on me", "I miss talking to my best friend"],
    "sleep": ["Work stress kept me up again last night", "I can't fall asleep because I'm thinking about work"],
    "dog": ["I really miss my dog", "It's been hard since we lost our dog"],
    "lonely": ["Bangalore still doesn't feel like home and I'm lonely", "I don't have anyone to talk to in this new city"],
    "guitar": ["I learned a new chord on the guitar today", "Guitar practice is going well"],
    "sister": ["Wedding preparations for my sister are getting busy", "What should I gift my sister for her wedding"],
    "running": ["My run this morning felt amazing", "I ran further than ever by the lake"],
    "offer": ["Should I accept the startup job in Pune", "I'm thinking about the Pune offer"],
    "exam": ["How should I prepare for my data structures exam", "Friday's exam is coming up fast"],
    "kitten": ["Mishti knocked my cup off the table today", "My kitten is growing so fast"],
    "cooking": ["I tried making biryani at home tonight", "Any easy recipes I could cook this week"],
    "trip": ["We booked the homestay in Darjeeling", "I'm excited for the Darjeeling trip"],
    "book": ["I've been thinking about what I read on stoicism", "That stoicism book changed how I see things"],
    "swim": ["I managed to float on my own in the pool today", "Swimming lessons are getting easier"],
}

UNRELATED = [
    "What a lovely sunny morning", "I'm excited for the weekend", "I just had the best coffee",
    "I watched a really fun movie last night", "Work was okay today, nothing special", "I'm thinking of starting a new hobby",
    "The rain today was so calming", "I cleaned my whole room and it feels great", "I tried a new cafe near my office",
    "My team won the quiz at work today", "I'm planning to repaint my room", "I had a nice long chat with my cousin",
    "Traffic was surprisingly light today", "I found a great new podcast", "I'm making tea and relaxing",
]

TANGENTIAL = {  # happy news that brushes a painful topic
    "dog": "My neighbour's new puppy is so adorable, I played with him all evening",
    "breakup": "My cousin got engaged today and I'm so happy for her",
    "panic": "My presentation went really well today, everyone clapped",
    "driving": "My brother passed his driving test today, we celebrated",
    "internship": "My friend got the internship she wanted and I'm proud of her",
    "parents": "My parents and I had a lovely dinner together tonight",
    "grandmother": "I baked cookies with my little niece today, it was so sweet",
    "sleep": "I slept nine hours last night and feel amazing",
    "lonely": "I made a new friend at the office today",
    "friend": "I had such a fun evening with my college friends",
}

FOLLOW_UPS = {
    "exam": ["How should I prepare for my data structures exam", "Which topics in data structures matter most", "I'm still revising trees and graphs for Friday"],
    "breakup": ["I miss my ex so much", "I keep checking her photos", "I don't know how to stop thinking about her"],
    "sleep": ["Work stress kept me up again", "I lay awake till three thinking about deadlines", "How do I switch off from work at night"],
}

PAINFUL = {k for k, _ in MEMORIES[:10]}


def methods(params: AgemParams):
    no_gate = dataclasses.replace(params, tau_painful=params.tau)
    no_refractory = dataclasses.replace(params, refractory_turns=0)
    return {
        "naive top-1 (original code)": ("naive", None),
        "top-3, cos >= 0.82 (progress report)": ("fixed", 0.82),
        f"top-3, cos >= {params.tau} (plain RAG)": ("fixed", params.tau),
        "AGEM without affect gate": ("agem", no_gate),
        "AGEM without refractory": ("agem", no_refractory),
        "AGEM (full)": ("agem", params),
    }


def select(kind, arg, q, qv, mem, mv, ma, age, last, turn, k):
    if kind == "naive":
        return naive_top1(q, mem)
    if kind == "fixed":
        return fixed_threshold(q, mem, k, arg)
    cands = rank(q, mem, mv, ma, age, turn - last, qv, arg)
    return [c.index for c in cands if c.gated is None]


def main():
    sense = AuraSense()
    keys = [k for k, _ in MEMORIES]

    def embed(texts):
        rs = [sense.read(t)[0] for t in texts]  # one sentence per call, as deployed
        return (np.vstack([r.embedding for r in rs]), np.array([r.affect.valence for r in rs], np.float32),
                np.array([r.affect.arousal for r in rs], np.float32))

    mem, mv, ma = embed([t for _, t in MEMORIES])
    age = np.random.default_rng(13).uniform(1, 60, len(MEMORIES))
    rel_q = [(k, q) for k, qs in RELEVANT.items() for q in qs]
    rq, rqv, _ = embed([q for _, q in rel_q])
    uq, uqv, _ = embed(UNRELATED)
    tq, tqv, _ = embed(list(TANGENTIAL.values()))

    params = AgemParams()
    never = np.full(len(MEMORIES), -10 ** 6)
    painful_idx = {keys.index(k) for k in PAINFUL}
    report = {
        "n": {"memories": len(MEMORIES), "painful": len(PAINFUL), "relevant_queries": len(rel_q),
              "unrelated_queries": len(UNRELATED), "tangential_queries": len(TANGENTIAL),
              "follow_up_turns": sum(len(v) - 1 for v in FOLLOW_UPS.values())},
        "memory_valence": {k: round(float(v), 3) for k, v in zip(keys, mv)},
        "painful_by_model": [k for k, v in zip(keys, mv) if v <= params.painful_valence],
        "params": dataclasses.asdict(params), "methods": {},
    }

    for name, (kind, arg) in methods(params).items():
        hits = [keys.index(k) in select(kind, arg, rq[i], rqv[i], mem, mv, ma, age, never, 0, params.k)
                for i, (k, _) in enumerate(rel_q)]
        inj_u = [select(kind, arg, uq[i], uqv[i], mem, mv, ma, age, never, 0, params.k) for i in range(len(UNRELATED))]
        inj_t = [select(kind, arg, tq[i], tqv[i], mem, mv, ma, age, never, 0, params.k) for i in range(len(TANGENTIAL))]
        isr_flags = [bool(painful_idx & set(s)) for s in inj_u + inj_t]
        tang_flags = [keys.index(k) in s for k, s in zip(TANGENTIAL, inj_t)]

        redundant = total = 0
        for turns in FOLLOW_UPS.values():
            fq, fqv, _ = embed(turns)
            last, prev = never.copy(), set()
            for t in range(len(turns)):
                chosen = set(select(kind, arg, fq[t], fqv[t], mem, mv, ma, age, last, t, params.k))
                if t > 0:
                    total += 1
                    redundant += bool(chosen & prev)
                for c in chosen:
                    last[c] = t
                prev = chosen

        report["methods"][name] = {
            "recall@3": round(float(np.mean(hits)), 4),
            "isr": round(float(np.mean(isr_flags)), 4),
            "tangential_target_surfaced": round(float(np.mean(tang_flags)), 4),
            "empty_rate_unrelated": round(float(np.mean([len(s) == 0 for s in inj_u])), 4),
            "mean_injected": round(float(np.mean([len(s) for s in inj_u + inj_t])), 2),
            "redundancy": round(redundant / max(total, 1), 4),
        }
        print(f"{name:38s} " + " ".join(f"{k}={v}" for k, v in report["methods"][name].items()))

    def demo(query):
        q, qv, _ = embed([query])
        cands = rank(q[0], mem, mv, ma, age, np.full(len(MEMORIES), 10 ** 6), float(qv[0]), params)
        top = sorted(cands, key=lambda c: -c.similarity)[:4]
        return {"query": query, "valence": round(float(qv[0]), 3),
                "items": [{"text": MEMORIES[c.index][1], "similarity": round(c.similarity, 3),
                           "admitted": c.gated is None, "gated": c.gated} for c in top]}

    report["demo"] = {"good": demo(TANGENTIAL["breakup"]), "hard": demo("I still miss her so much, the breakup hurts tonight")}
    base = report["methods"][f"top-3, cos >= {params.tau} (plain RAG)"]
    ours = report["methods"]["AGEM (full)"]
    if base["isr"] > 0:
        report["isr_reduction_vs_plain_rag"] = round(1 - ours["isr"] / base["isr"], 3)
    (RESULTS / "agem_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("painful by model valence:", report["painful_by_model"])
    print(json.dumps(report["demo"], indent=1))


if __name__ == "__main__":
    main()
