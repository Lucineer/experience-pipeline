"""Experience Pipeline — Training data from git history.

Captures agent-environment interactions as Experience Primitives,
curates quality locally, does LoRA fine-tuning on-device,
aggregates across fleet via federated learning.

The repository IS the training dataset.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
import time
import json
import uuid
import hashlib


class ExperienceTier(Enum):
    TIER_1 = 1  # High quality — auto-train
    TIER_2 = 2  # Medium — human review needed
    TIER_3 = 3  # Low — archive only


class ExperienceSource(Enum):
    SENSOR = "sensor"
    ACTION = "action"
    HUMAN_FEEDBACK = "human_feedback"
    SYSTEM_EVENT = "system_event"
    FLEET_SYNC = "fleet_sync"


@dataclass
class ExperiencePrimitive:
    """Atomic unit of learning data."""
    experience_id: str = ""
    timestamp: float = 0.0
    vessel_id: str = ""
    agent_version: str = ""

    # Context
    environment_state: Dict[str, Any] = field(default_factory=dict)
    task_specification: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    human_intent: str = ""

    # Observations
    sensor_data: List[Dict[str, float]] = field(default_factory=list)
    perception_outputs: Dict[str, Any] = field(default_factory=dict)
    internal_state: Dict[str, Any] = field(default_factory=dict)

    # Actions
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)

    # Outcomes
    outcome: str = ""  # success, partial, failure
    reward: float = 0.0
    learned: str = ""

    # Metadata
    source: ExperienceSource = ExperienceSource.SENSOR
    tier: ExperienceTier = ExperienceTier.TIER_2
    quality_score: float = 0.0
    commit_sha: str = ""
    git_diff: str = ""

    def __post_init__(self):
        if not self.experience_id:
            self.experience_id = uuid.uuid4().hex[:16]
        if not self.timestamp:
            self.timestamp = time.time()

    def content_hash(self) -> str:
        data = json.dumps({
            "ts": self.timestamp, "v": self.vessel_id,
            "env": self.environment_state, "outcome": self.outcome,
            "reward": self.reward, "learned": self.learned,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def is_duplicate(self, other: "ExperiencePrimitive") -> bool:
        return self.content_hash() == other.content_hash()

    def to_dict(self) -> dict:
        return {
            "id": self.experience_id,
            "ts": self.timestamp,
            "vessel": self.vessel_id,
            "ver": self.agent_version,
            "env": self.environment_state,
            "task": self.task_specification,
            "sensors": self.sensor_data[:3],  # Truncate for storage
            "perception": self.perception_outputs,
            "actions": self.actions_taken[:3],
            "reasoning": self.reasoning_trace[:5],
            "outcome": self.outcome,
            "reward": self.reward,
            "learned": self.learned,
            "source": self.source.value,
            "tier": self.tier.value,
            "quality": self.quality_score,
            "sha": self.commit_sha,
            "hash": self.content_hash(),
        }

    def to_training_format(self) -> Dict[str, Any]:
        """Convert to format suitable for LoRA fine-tuning."""
        system = self.task_specification.get("system_prompt", "")
        user_input = self._context_to_prompt()
        assistant_output = self._outcome_to_response()
        return {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_output},
            ],
            "weight": self.quality_score,
            "tier": self.tier.value,
        }

    def _context_to_prompt(self) -> str:
        parts = [f"Environment: {json.dumps(self.environment_state)[:200]}"]
        if self.sensor_data:
            parts.append(f"Sensors: {json.dumps(self.sensor_data[0])}")
        if self.constraints:
            parts.append(f"Constraints: {json.dumps(self.constraints)}")
        if self.human_intent:
            parts.append(f"Human intent: {self.human_intent}")
        return "\n".join(parts)

    def _outcome_to_response(self) -> str:
        parts = []
        if self.reasoning_trace:
            parts.extend(self.reasoning_trace[:5])
        if self.actions_taken:
            for a in self.actions_taken[:3]:
                parts.append(f"Action: {a.get('type', '?')} -> {a.get('result', '?')}")
        parts.append(f"Outcome: {self.outcome} (reward={self.reward})")
        if self.learned:
            parts.append(f"Learned: {self.learned}")
        return "\n".join(parts)


class ExperienceBuffer:
    """Local experience capture and curation buffer."""

    def __init__(self, vessel_id: str, max_size: int = 10000):
        self.vessel_id = vessel_id
        self.max_size = max_size
        self._buffer: List[ExperiencePrimitive] = []
        self._seen_hashes: set = set()
        self._tier_counts = {1: 0, 2: 0, 3: 0}

    def capture(self, experience: ExperiencePrimitive) -> bool:
        if experience.content_hash() in self._seen_hashes:
            return False
        if len(self._buffer) >= self.max_size:
            self._buffer.pop(0)
        experience.vessel_id = self.vessel_id
        self._buffer.append(experience)
        self._seen_hashes.add(experience.content_hash())
        self._tier_counts[experience.tier.value] += 1
        return True

    def curate(self, quality_fn: Optional[Callable] = None):
        """Score and tier experiences by quality."""
        for exp in self._buffer:
            if quality_fn:
                exp.quality_score = quality_fn(exp)
            else:
                exp.quality_score = self._default_quality(exp)
            if exp.quality_score >= 0.8:
                exp.tier = ExperienceTier.TIER_1
            elif exp.quality_score >= 0.5:
                exp.tier = ExperienceTier.TIER_2
            else:
                exp.tier = ExperienceTier.TIER_3
        self._tier_counts = {1: 0, 2: 0, 3: 0}
        for exp in self._buffer:
            self._tier_counts[exp.tier.value] += 1

    def _default_quality(self, exp: ExperiencePrimitive) -> float:
        score = 0.0
        if exp.outcome == "success":
            score += 0.3
        elif exp.outcome == "partial":
            score += 0.15
        score += min(exp.reward / 10.0, 0.3)
        if exp.learned:
            score += 0.2
        if exp.human_intent:
            score += 0.1
        if exp.reasoning_trace:
            score += min(len(exp.reasoning_trace) * 0.02, 0.1)
        return min(score, 1.0)

    def get_training_batch(self, tier: ExperienceTier = ExperienceTier.TIER_1,
                           limit: int = 100) -> List[Dict[str, Any]]:
        eligible = [e for e in self._buffer if e.tier.value <= tier.value]
        eligible.sort(key=lambda e: e.quality_score, reverse=True)
        return [e.to_training_format() for e in eligible[:limit]]

    def get_tier1_count(self) -> int:
        return self._tier_counts[1]

    def stats(self) -> Dict[str, Any]:
        outcomes = {}
        for e in self._buffer:
            outcomes[e.outcome] = outcomes.get(e.outcome, 0) + 1
        return {
            "vessel_id": self.vessel_id,
            "total": len(self._buffer),
            "unique_hashes": len(self._seen_hashes),
            "tiers": self._tier_counts,
            "outcomes": outcomes,
            "avg_quality": round(
                sum(e.quality_score for e in self._buffer) / max(1, len(self._buffer)), 2),
        }

    def export_for_federation(self) -> Dict[str, Any]:
        """Export Tier 1 experiences for fleet aggregation."""
        tier1 = [e.to_dict() for e in self._buffer if e.tier.value == 1]
        return {
            "vessel_id": self.vessel_id,
            "count": len(tier1),
            "experiences": tier1,
            "exported_at": time.time(),
        }
