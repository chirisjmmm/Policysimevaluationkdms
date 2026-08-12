"""Common intermediate representation (sec 2.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

PHASE_ORDER = ["Inputs", "Activities", "Outputs", "Outcomes", "Impact"]
PHASE_INDEX = {p: i for i, p in enumerate(PHASE_ORDER)}


@dataclass(frozen=True)
class Post:
    cell_id: str
    run_id: str
    path_key: str
    phase: str
    phase_index: int
    order_index: int
    agent_key: str                 # persona_name
    stakeholder_type: str
    role_slot: str                 # "AIResearcher#1" -- disambiguates duplicate stakeholder types
    prediction_values: dict
    narrative: str
    evidence: tuple
    transition_mode: Optional[str] = None


@dataclass(frozen=True)
class PhaseState:
    phase: str
    phase_index: int
    posts: tuple
    phase_summary: str
    posting_order: tuple
    cross_checks: dict = field(default_factory=dict)   # variable -> raw cross_check record


@dataclass(frozen=True)
class Run:
    cell_id: str
    run_id: str
    source: str                     # "single" | "branch"
    transition_path: tuple
    phases: tuple
    target_values: dict             # Outcomes-phase aggregated_value per variable (this run's own prediction, not GT)
    participants: tuple
    scenario_impact: dict

    def phase(self, name: str) -> PhaseState:
        for p in self.phases:
            if p.phase == name:
                return p
        raise KeyError(name)

    def all_posts(self):
        for ph in self.phases:
            for post in ph.posts:
                yield post
