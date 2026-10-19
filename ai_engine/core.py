from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .learning import QLearner
from .planning import GoapPlanner, PlanResult


StateDict = Dict[str, Any]


@dataclass(frozen=True)
class WorldState:
    data: StateDict

    def satisfies(self, goal: StateDict) -> bool:
        return all(self.data.get(k) == v for k, v in goal.items())

    def apply(self, effects: StateDict) -> "WorldState":
        new_data = dict(self.data)
        new_data.update(effects)
        return WorldState(new_data)


@dataclass(frozen=True)
class Action:
    name: str
    cost: float
    preconditions: StateDict
    effects: StateDict
    utility: Optional[Callable[[WorldState], float]] = None

    def is_applicable(self, state: WorldState) -> bool:
        return all(state.data.get(k) == v for k, v in self.preconditions.items())

    def score(self, state: WorldState) -> float:
        if self.utility is None:
            return 0.0
        return float(self.utility(state))


@dataclass
class GameAI:
    """Hybrid AI combining planning, utility selection, and Q-learning."""

    actions: List[Action]
    planner: GoapPlanner = field(default_factory=GoapPlanner)
    learner: QLearner = field(default_factory=QLearner)
    exploration: float = 0.1

    def plan(self, start: WorldState, goal: StateDict) -> PlanResult:
        return self.planner.plan(start, goal, self.actions)

    def choose_action(self, state: WorldState) -> Action:
        applicable = [action for action in self.actions if action.is_applicable(state)]
        if not applicable:
            raise ValueError("No applicable actions for current state")

        learned = self.learner.best_action(state, applicable, self.exploration)
        if learned is not None:
            return learned

        return max(applicable, key=lambda action: action.score(state))

    def learn(self, state: WorldState, action: Action, reward: float, next_state: WorldState) -> None:
        self.learner.update(state, action, reward, next_state)

    def export_policy(self, state_samples: Iterable[WorldState]) -> Dict[str, str]:
        """Return a dict mapping serialized states to best action names for Unity import."""
        policy = {}
        for sample in state_samples:
            best = self.choose_action(sample)
            policy[self.learner.state_key(sample)] = best.name
        return policy
