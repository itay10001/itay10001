from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import heapq

from .core import Action, WorldState


@dataclass
class PlanResult:
    success: bool
    actions: List[Action]
    cost: float


class GoapPlanner:
    """Simple GOAP planner using A* search over world states."""

    def plan(self, start: WorldState, goal: Dict[str, object], actions: Iterable[Action]) -> PlanResult:
        open_set: List[Tuple[float, int, WorldState, List[Action], float]] = []
        visited = set()
        counter = 0

        heapq.heappush(open_set, (0.0, counter, start, [], 0.0))

        while open_set:
            _, _, state, path, cost_so_far = heapq.heappop(open_set)
            state_key = tuple(sorted(state.data.items()))
            if state_key in visited:
                continue
            visited.add(state_key)

            if state.satisfies(goal):
                return PlanResult(True, path, cost_so_far)

            for action in actions:
                if not action.is_applicable(state):
                    continue
                next_state = state.apply(action.effects)
                next_cost = cost_so_far + action.cost
                heuristic = self._heuristic(next_state, goal)
                counter += 1
                heapq.heappush(open_set, (next_cost + heuristic, counter, next_state, path + [action], next_cost))

        return PlanResult(False, [], float("inf"))

    @staticmethod
    def _heuristic(state: WorldState, goal: Dict[str, object]) -> float:
        mismatches = sum(1 for k, v in goal.items() if state.data.get(k) != v)
        return float(mismatches)
