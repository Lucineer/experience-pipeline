# experience-pipeline

Training data pipeline from git history. Captures agent-environment interactions, curates quality, formats for LoRA fine-tuning, and federates across the fleet.

## The Pipeline

```
Interaction → Experience Primitive → Quality Score → Tier Assignment
                                                    ├─ Tier 1: Auto-train
                                                    ├─ Tier 2: Human review
                                                    └─ Tier 3: Archive
                                                                  ↓
                                                    LoRA Training Format
                                                                  ↓
                                                    Federated Aggregation
```

## Usage

```python
from pipeline import ExperienceBuffer, ExperiencePrimitive, ExperienceSource, ExperienceTier

buffer = ExperienceBuffer("vessel-01")

# Capture an experience
exp = ExperiencePrimitive(
    vessel_id="vessel-01",
    environment_state={"speed": 5.2, "heading": 270, "depth": 12},
    task_specification={"name": "dock_approach"},
    actions_taken=[{"type": "throttle", "result": "reduced_to_2_knots"}],
    outcome="success",
    reward=8.5,
    learned="Reduce speed below 3 knots within 50m of dock",
    source=ExperienceSource.SENSOR)

buffer.capture(exp)

# Curate and score
buffer.curate()

# Export for LoRA training
batch = buffer.get_training_batch(ExperienceTier.TIER_1, limit=50)
# Each entry: {"messages": [system, user, assistant], "weight": 0.9}

# Federate to fleet
federation_data = buffer.export_for_federation()
```

## Quality Scoring

- Success outcome: +0.3
- Partial outcome: +0.15
- Reward: up to +0.3
- Has "learned" field: +0.2
- Has human intent: +0.1
- Reasoning trace: up to +0.1
- **Tier 1**: score >= 0.8
- **Tier 2**: score >= 0.5
- **Tier 3**: score < 0.5

Part of the [Lucineer ecosystem](https://github.com/Lucineer/the-fleet).
