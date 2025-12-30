"""Generate EditPlan objects for multicam angle selections."""

from __future__ import annotations

from typing import Dict, List, Optional

from VideoForge.agent.planner import EditAction, EditPlan


class MulticamPlanGenerator:
    """Convert angle selections into EditPlan actions."""

    def generate_plan(
        self,
        angle_selections: List[Dict],
        timeline_info: Dict,
        source_clips: List[str],
        source_paths: Optional[List[str]] = None,
    ) -> EditPlan:
        actions: List[EditAction] = []
        fps = timeline_info.get("fps", 24.0)
        try:
            fps = float(fps)
        except Exception:
            fps = 24.0

        for sel in angle_selections:
            start_sec = float(sel.get("start_sec", 0.0))
            cut_frame = int(start_sec * fps)
            angle_index = int(sel.get("angle_index", 0))
            actions.append(
                EditAction(
                    action_type="cut_and_switch_angle",
                    target=f"V{angle_index + 1}",
                    params={
                        "cut_frame": cut_frame,
                        "target_angle_track": angle_index,
                        "start_sec": start_sec,
                        "end_sec": float(sel.get("end_sec", 0.0)),
                        "source_clips": source_clips,
                        "source_paths": source_paths or [],
                        "source_path": sel.get("source_path"),
                        "source_in_sec": sel.get("source_in_sec"),
                        "source_out_sec": sel.get("source_out_sec"),
                        "source_ranges": sel.get("source_ranges", []),
                        "angle_score": float(sel.get("angle_score", 0.0)),
                    },
                    reason=str(sel.get("reason", "") or "Auto multicam selection"),
                )
            )

        unique_angles = len({int(sel.get("angle_index", 0)) for sel in angle_selections}) if angle_selections else 0
        avg_score = 0.0
        if angle_selections:
            avg_score = sum(float(sel.get("angle_score", 0.0)) for sel in angle_selections) / len(
                angle_selections
            )

        return EditPlan(
            request="Auto Multicam Cut",
            summary=(
                f"Generated {len(actions)} cuts across {unique_angles} angles "
                f"(avg score: {avg_score:.2f})"
            ),
            actions=actions,
            estimated_clips=len(actions),
            risk_level="moderate",
            warnings=[
                "This will split and rearrange multicam tracks",
                "Make sure all angles are synced before applying",
            ],
        )
