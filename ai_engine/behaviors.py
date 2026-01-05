from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List


class Status(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class Node:
    def tick(self) -> Status:
        raise NotImplementedError


@dataclass
class Condition(Node):
    predicate: Callable[[], bool]

    def tick(self) -> Status:
        return Status.SUCCESS if self.predicate() else Status.FAILURE


@dataclass
class ActionNode(Node):
    action: Callable[[], bool]

    def tick(self) -> Status:
        return Status.SUCCESS if self.action() else Status.FAILURE


@dataclass
class Sequence(Node):
    children: List[Node]

    def tick(self) -> Status:
        for child in self.children:
            result = child.tick()
            if result != Status.SUCCESS:
                return result
        return Status.SUCCESS


@dataclass
class Selector(Node):
    children: List[Node]

    def tick(self) -> Status:
        for child in self.children:
            result = child.tick()
            if result == Status.SUCCESS:
                return Status.SUCCESS
        return Status.FAILURE


@dataclass
class BehaviorTree:
    root: Node

    def tick(self) -> Status:
        return self.root.tick()
