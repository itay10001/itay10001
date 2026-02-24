from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from .core import Action, WorldState


@dataclass
class QLearner:
    """Tabular Q-learning for discrete state/action spaces."""

    alpha: float = 0.3
    gamma: float = 0.9
    q_table: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def state_key(self, state: WorldState) -> str:
        return "|".join(f"{k}={state.data.get(k)}" for k in sorted(state.data.keys()))

    def best_action(self, state: WorldState, actions: Iterable[Action], exploration: float) -> Optional[Action]:
        if random.random() < exploration:
            return random.choice(list(actions))

        key = self.state_key(state)
        if key not in self.q_table:
            return None

        action_scores = self.q_table[key]
        if not action_scores:
            return None

        best_name = max(action_scores, key=action_scores.get)
        for action in actions:
            if action.name == best_name:
                return action
        return None

    def update(self, state: WorldState, action: Action, reward: float, next_state: WorldState) -> None:
        state_key = self.state_key(state)
        next_key = self.state_key(next_state)

        self.q_table.setdefault(state_key, {})
        self.q_table.setdefault(next_key, {})

        current = self.q_table[state_key].get(action.name, 0.0)
        next_best = max(self.q_table[next_key].values(), default=0.0)
        target = reward + self.gamma * next_best
        self.q_table[state_key][action.name] = current + self.alpha * (target - current)
