"""AGEM: Affect-Gated Episodic Memory.

Plain retrieval-augmented memory injects whatever is most similar. For a companion
that is wrong in two ways the NeuroCare progress report already observed: it keeps
resurfacing painful episodes on good days, and it repeats the same memory turn after
turn. AGEM ranks and gates memories with four signals:

  relevance     cosine similarity of the query and memory embeddings (Aura-Sense)
  salience      emotional intensity of the memory when it was stored
  recency       exponential decay with a configurable half-life
  refractory    a memory surfaced in the last R turns is suppressed

and one gate:

  affect gate   a negative-valence memory may surface only when the query is
                strongly related to it, unless the person is currently in a
                negative state themselves. Painful history stays available when
                it is the topic, and stays quiet when it is not.

Everything here is pure numpy so the benchmark in research/ evaluates this exact code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgemParams:
    k: int = 3
    tau: float = 0.38            # base relevance gate
    tau_painful: float = 0.55    # relevance needed for a painful memory when the person is not low
    painful_valence: float = -0.12
    beta_salience: float = 0.25
    gamma_recency: float = 0.05
    half_life_days: float = 21.0
    refractory_turns: int = 3
    refractory_penalty: float = 0.6


@dataclass
class Candidate:
    index: int
    score: float
    similarity: float
    gated: str | None  # None if admitted, else the reason it was held back


def salience(valence: np.ndarray, arousal: np.ndarray) -> np.ndarray:
    return np.clip(np.sqrt(valence ** 2 + np.clip(arousal, 0, None) ** 2), 0, 1)


def rank(query: np.ndarray, mem: np.ndarray, mem_valence: np.ndarray, mem_arousal: np.ndarray,
         age_days: np.ndarray, turns_since_surfaced: np.ndarray, query_valence: float,
         params: AgemParams = AgemParams()) -> list[Candidate]:
    """Admitted memories (best first) followed by a few gated ones, for explainability.

    turns_since_surfaced holds a large number for memories never surfaced.
    """
    if len(mem) == 0:
        return []
    sim = mem @ query
    sal = salience(mem_valence, mem_arousal)
    rec = np.exp(-math.log(2) * age_days / params.half_life_days)
    score = sim * (1 + params.beta_salience * sal) + params.gamma_recency * rec
    refractory = turns_since_surfaced < params.refractory_turns
    score = np.where(refractory, score - params.refractory_penalty, score)

    person_low = query_valence <= params.painful_valence
    out = []
    for i in np.argsort(-score):
        reason = None
        if sim[i] < params.tau:
            reason = "not relevant"
        elif refractory[i]:
            reason = "recently surfaced"
        elif mem_valence[i] <= params.painful_valence and not person_low and sim[i] < params.tau_painful:
            reason = "painful and only loosely related"
        out.append(Candidate(int(i), float(score[i]), float(sim[i]), reason))
    admitted = [c for c in out if c.gated is None][: params.k]
    held = [c for c in out if c.gated is not None and c.similarity >= params.tau * 0.8][:3]
    return admitted + held


def naive_top1(query: np.ndarray, mem: np.ndarray) -> list[int]:
    """The original NeuroCare behaviour: always inject the single nearest memory."""
    return [int(np.argmax(mem @ query))] if len(mem) else []


def fixed_threshold(query: np.ndarray, mem: np.ndarray, k: int, tau: float) -> list[int]:
    if not len(mem):
        return []
    sim = mem @ query
    idx = np.argsort(-sim)[:k]
    return [int(i) for i in idx if sim[i] >= tau]
