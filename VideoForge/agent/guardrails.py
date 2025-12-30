"""Guardrails for agent actions."""

from __future__ import annotations

from typing import List, Tuple

from VideoForge.agent.planner import EditAction, EditPlan
from VideoForge.config.config_manager import Config


def check_destructive_action(action: EditAction) -> Tuple[bool, str]:
    """Return whether an action is destructive and a warning message."""
    destructive_types = {"delete_clip", "move_clip", "cut_clip"}
    if action.action_type in destructive_types:
        return True, f"WARNING: Destructive action: {action.action_type}"
    return False, ""


def validate_plan_safety(plan: EditPlan) -> List[str]:
    """Return guardrail warnings for a plan."""
    warnings: List[str] = []
    destructive_count = 0
    for action in plan.actions:
        destructive, _ = check_destructive_action(action)
        if destructive:
            destructive_count += 1
    if destructive_count:
        warnings.append(f"{destructive_count} destructive actions detected")

    max_actions = int(Config.get("agent_max_auto_actions", 5))
    if len(plan.actions) > max_actions:
        warnings.append(
            f"Plan has {len(plan.actions)} actions (max: {max_actions})"
        )
    return warnings

