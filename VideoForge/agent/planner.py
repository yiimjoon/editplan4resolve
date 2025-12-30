"""Edit plan schema and helpers for VideoForge agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

ActionType = Literal[
    "insert_clip",
    "delete_clip",
    "move_clip",
    "cut_clip",
    "sync_clips",
    "match_broll",
    "apply_transition",
]

RiskLevel = Literal["safe", "moderate", "destructive"]


@dataclass
class EditAction:
    """Single editing action."""

    action_type: ActionType
    target: str
    params: Dict[str, Any]
    reason: str


@dataclass
class EditPlan:
    """Full editing plan proposed by an agent."""

    request: str
    summary: str
    actions: List[EditAction]
    estimated_clips: int
    risk_level: RiskLevel
    warnings: List[str]


def edit_plan_from_json(payload: str) -> EditPlan:
    """Parse a JSON string into an EditPlan."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("EditPlan payload is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError("EditPlan payload must be a JSON object.")
    return edit_plan_from_dict(data)


def edit_plan_from_dict(data: Dict[str, Any]) -> EditPlan:
    """Parse a dictionary into an EditPlan."""
    if not isinstance(data, dict):
        raise ValueError("EditPlan payload must be a JSON object.")
    actions_raw = data.get("actions")
    if not isinstance(actions_raw, list):
        raise ValueError("EditPlan.actions must be a list.")

    actions: List[EditAction] = []
    skipped_actions: List[str] = []
    for action in actions_raw:
        if not isinstance(action, dict):
            raise ValueError("EditPlan.actions entries must be objects.")
        raw_action_type = str(
            action.get("action_type")
            or action.get("action")
            or action.get("type")
            or ""
        ).strip()
        action_type = _normalize_action_type(raw_action_type)
        if action_type is None:
            skipped_actions.append(raw_action_type)
            continue
        if action_type not in _action_type_values():
            raise ValueError(f"Invalid action_type: {action_type}")
        params = action.get("params")
        if not isinstance(params, dict):
            params = {}
        target = str(action.get("target", "") or "").strip()
        if not target:
            if action_type in {"delete_clip", "move_clip", "cut_clip"}:
                target = "playhead"
            elif action_type == "insert_clip":
                target = "V1"
        actions.append(
            EditAction(
                action_type=action_type,
                target=target,
                params=params,
                reason=str(action.get("reason", "") or ""),
            )
        )

    if not actions:
        raise ValueError("EditPlan.actions is empty.")

    request = str(data.get("request", "") or "").strip()
    summary = str(data.get("summary", "") or "").strip()
    if not summary:
        summary = "Planned edits based on the latest request."

    estimated_clips = data.get("estimated_clips")
    if estimated_clips is None:
        estimated_clips = len(actions)
    try:
        estimated_clips = int(estimated_clips)
    except Exception:
        estimated_clips = len(actions)

    risk_level = data.get("risk_level")
    if risk_level not in _risk_level_values():
        destructive = {"delete_clip", "move_clip", "cut_clip"}
        if any(action.action_type in destructive for action in actions):
            risk_level = "destructive"
        else:
            risk_level = "safe"

    warnings = data.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    warnings_list = [str(item) for item in warnings]
    if skipped_actions:
        warnings_list.append(
            f"Removed non-edit actions: {', '.join(skipped_actions)}"
        )

    return EditPlan(
        request=request,
        summary=summary,
        actions=actions,
        estimated_clips=estimated_clips,
        risk_level=risk_level,
        warnings=warnings_list,
    )


def edit_plan_summary(plan: EditPlan) -> str:
    """Return a concise summary for user approval."""
    count = len(plan.actions)
    warn = f" Warnings: {len(plan.warnings)}." if plan.warnings else ""
    return f"{plan.summary} (Actions: {count}, Risk: {plan.risk_level}).{warn}"


def validate_edit_plan(plan: EditPlan) -> List[str]:
    """Return validation issues for a plan."""
    issues: List[str] = []
    if not plan.request:
        issues.append("Missing request text.")
    if not plan.summary:
        issues.append("Missing summary.")
    if plan.risk_level not in _risk_level_values():
        issues.append(f"Invalid risk level: {plan.risk_level}")
    if plan.estimated_clips < 0:
        issues.append("Estimated clips cannot be negative.")
    if not plan.actions:
        issues.append("No actions provided.")
    for idx, action in enumerate(plan.actions, start=1):
        if action.action_type not in _action_type_values():
            issues.append(f"Action {idx}: invalid action_type {action.action_type}")
        if not action.target:
            issues.append(f"Action {idx}: missing target")
        if not isinstance(action.params, dict):
            issues.append(f"Action {idx}: params must be an object")
        if not action.reason:
            issues.append(f"Action {idx}: missing reason")
    return issues


def _require_keys(data: Dict[str, Any], keys: List[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(missing)}")


def _action_type_values() -> List[str]:
    return [
        "insert_clip",
        "delete_clip",
        "move_clip",
        "cut_clip",
        "sync_clips",
        "match_broll",
        "apply_transition",
    ]


def _risk_level_values() -> List[str]:
    return ["safe", "moderate", "destructive"]


def _normalize_action_type(action_type: str) -> str | None:
    """Normalize tool-style action names into EditPlan action types."""
    aliases = {
        "delete_clip_at_playhead": "delete_clip",
        "sync_clips_from_timeline": "sync_clips",
        "match_broll_for_segment": "match_broll",
        "insert_clip": "insert_clip",
        "get_timeline_info": None,
        "get_clips_at_playhead": None,
    }
    if action_type in aliases:
        return aliases[action_type]
    return action_type if action_type else None

