"""Runtime and state models for the VideoForge agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from VideoForge.agent.planner import EditPlan

AgentMode = Literal["recommend_only", "approve_required", "full_access"]


@dataclass
class AgentRuntime:
    """Static runtime context for a session."""

    user_id: str = "default_user"
    project_key: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentMessage:
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


@dataclass
class AgentState:
    """Dynamic state for agent execution."""

    session_id: str
    mode: AgentMode
    conversation: List[AgentMessage] = field(default_factory=list)
    pending_plan: Optional[EditPlan] = None
    pending_tool_calls: Optional[List[Dict[str, Any]]] = None
    timeline_context: Optional[Dict[str, Any]] = None

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self.conversation.append(AgentMessage(role=role, content=content, **kwargs))

