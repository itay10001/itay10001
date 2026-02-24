from __future__ import annotations

import json
import random

from .core import Action, GameAI, WorldState


def main() -> None:
    actions = [
        Action(
            name="gather_wood",
            cost=1.0,
            preconditions={},
            effects={"wood": 1},
            utility=lambda state: 5.0 if state.data.get("wood", 0) == 0 else 1.0,
        ),
        Action(
            name="craft_axe",
            cost=2.0,
            preconditions={"wood": 1},
            effects={"has_axe": True},
            utility=lambda state: 3.0 if not state.data.get("has_axe") else 0.0,
        ),
        Action(
            name="collect_stone",
            cost=1.5,
            preconditions={"has_axe": True},
            effects={"stone": 1},
            utility=lambda state: 4.0 if state.data.get("stone", 0) == 0 else 1.0,
        ),
    ]

    ai = GameAI(actions=actions)

    state = WorldState({"wood": 0, "stone": 0, "has_axe": False})
    goal = {"stone": 1}

    plan = ai.plan(state, goal)
    print("Plan:", [action.name for action in plan.actions], "cost=", plan.cost)

    for _ in range(5):
        action = ai.choose_action(state)
        reward = random.uniform(0.0, 1.0)
        next_state = state.apply(action.effects)
        ai.learn(state, action, reward, next_state)
        state = next_state

    policy = ai.export_policy([state])
    print("Exported policy:")
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
