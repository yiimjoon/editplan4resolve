"""Agent UI section for VideoForge."""

from __future__ import annotations

import json
import uuid
from typing import Optional

from VideoForge.agent.executor import AgentExecutor
from VideoForge.agent.planner import EditPlan, edit_plan_summary
from VideoForge.agent.runtime import AgentState
from VideoForge.agent.tools import TOOL_REGISTRY, get_tool_schema
from VideoForge.ai.llm_hooks import get_llm_api_key, is_llm_enabled
from VideoForge.config.config_manager import Config
from VideoForge.ui.qt_compat import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QTextBrowser,
    QThread,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)


class AgentWorker(QThread):
    response_ready = Signal(str)
    plan_ready = Signal(object)
    tool_calls_ready = Signal(object)

    def __init__(self, executor: AgentExecutor, state: AgentState, user_input: str) -> None:
        super().__init__()
        self.executor = executor
        self.state = state
        self.user_input = user_input

    def run(self) -> None:
        try:
            response = self.executor.run(self.state, self.user_input)
            if response:
                self.response_ready.emit(response)
            if self.state.pending_plan:
                self.plan_ready.emit(self.state.pending_plan)
            if self.state.pending_tool_calls:
                self.tool_calls_ready.emit(self.state.pending_tool_calls)
        except Exception as exc:
            self.response_ready.emit(f"Error: {exc}")


class AgentSection(QWidget):
    """Agent chat panel with approval flow."""

    def __init__(self, resolve_api=None) -> None:
        super().__init__()
        self.resolve_api = resolve_api
        self.executor: Optional[AgentExecutor] = None
        self.state: Optional[AgentState] = None
        self.worker: Optional[AgentWorker] = None
        self._init_ui()
        self._load_config()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        header.setSpacing(12)

        mode_label = QLabel("Agent Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["recommend_only", "approve_required", "full_access"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)

        self.status_label = QLabel("API Key Status: Not connected")
        self.status_label.setStyleSheet("font-size: 11px;")

        header.addWidget(mode_label)
        header.addWidget(self.mode_combo)
        header.addStretch()
        header.addWidget(self.status_label)
        layout.addLayout(header)

        self.chat_display = QTextBrowser()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("background: white; font-size: 12px;")
        layout.addWidget(self.chat_display, 1)

        self.plan_frame = QFrame()
        self.plan_frame.setFrameShape(QFrame.StyledPanel)
        plan_layout = QVBoxLayout(self.plan_frame)
        self.plan_summary = QLabel("Pending plan")
        self.plan_summary.setStyleSheet("font-size: 11px;")
        plan_layout.addWidget(self.plan_summary)

        self.plan_actions = QListWidget()
        plan_layout.addWidget(self.plan_actions)

        plan_buttons = QHBoxLayout()
        self.plan_approve_btn = QPushButton("Approve")
        self.plan_reject_btn = QPushButton("Reject")
        self.plan_modify_btn = QPushButton("Modify")
        self.plan_approve_btn.clicked.connect(self.on_approve_plan)
        self.plan_reject_btn.clicked.connect(self.on_reject_plan)
        self.plan_modify_btn.clicked.connect(self.on_modify_plan)
        plan_buttons.addWidget(self.plan_approve_btn)
        plan_buttons.addWidget(self.plan_reject_btn)
        plan_buttons.addWidget(self.plan_modify_btn)
        plan_layout.addLayout(plan_buttons)

        self.plan_frame.setVisible(False)
        layout.addWidget(self.plan_frame)

        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a request for the agent...")
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send_message)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.on_clear_chat)
        self.load_timeline_btn = QPushButton("Load Timeline Info")
        self.load_timeline_btn.clicked.connect(self.on_load_timeline_info)

        input_row.addWidget(self.input_field, 1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.clear_btn)
        input_row.addWidget(self.load_timeline_btn)
        layout.addLayout(input_row)

    def _load_config(self) -> None:
        if not is_llm_enabled():
            self._update_status("API Key Status: Disabled (LLM off)")
            self.mode_combo.setEnabled(False)
            return
        api_key = str(Config.get("agent_api_key", "")).strip() or (get_llm_api_key() or "")
        mode = str(Config.get("agent_mode", "approve_required")).strip()
        if mode in {"recommend_only", "approve_required", "full_access"}:
            self.mode_combo.setCurrentText(mode)
        if api_key:
            self._init_executor(api_key)
        else:
            self._update_status("API Key Status: Missing")

    def _init_executor(self, api_key: str) -> None:
        try:
            self.executor = AgentExecutor(api_key=api_key)
            self.state = AgentState(session_id=str(uuid.uuid4()), mode=self.mode_combo.currentText())
            self._update_status("API Key Status: Connected")
        except Exception as exc:
            self.executor = None
            self.state = None
            self._update_status(f"API Key Status: Error ({exc})")

    def _update_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _on_mode_changed(self, value: str) -> None:
        Config.set("agent_mode", value)
        if self.state:
            self.state.mode = value

    def on_send_message(self) -> None:
        text = self.input_field.text().strip()
        if not text:
            return
        if not is_llm_enabled():
            self.append_chat("system", "Agent disabled: enable LLM provider in Settings.")
            return
        if not self.executor or not self.state:
            api_key = str(Config.get("agent_api_key", "")).strip() or (get_llm_api_key() or "")
            if api_key:
                self._init_executor(api_key)
            if not self.executor or not self.state:
                self.append_chat("system", "Agent not ready. Set API key in Settings.")
                return

        self.append_chat("user", text)
        self.input_field.clear()
        self._run_agent(text)

    def on_agent_response(self, text: str) -> None:
        self.append_chat("assistant", text)

    def on_plan_ready(self, plan: EditPlan) -> None:
        self.state.pending_plan = plan
        if self.state and self.state.mode == "full_access":
            self.on_approve_plan()
            return
        self.show_plan_approval_widget(plan)

    def on_approve_plan(self) -> None:
        plan = self.state.pending_plan if self.state else None
        if not plan:
            return
        self.append_chat("system", f"Executing {len(plan.actions)} actions...")
        results = []
        for action in plan.actions:
            result = self._execute_action(action)
            results.append(result)
            self.append_chat("system", f"{action.action_type}: {result}")
        self.state.pending_plan = None
        self.hide_plan_approval_widget()
        self.append_chat("system", "Plan execution complete.")

    def on_reject_plan(self) -> None:
        if self.state:
            self.state.pending_plan = None
        self.hide_plan_approval_widget()
        self.append_chat("system", "Plan rejected. Provide new instructions.")

    def on_modify_plan(self) -> None:
        if self.state:
            self.state.pending_plan = None
        self.hide_plan_approval_widget()
        self.append_chat("system", "Plan cleared. Describe the changes you want.")

    def on_clear_chat(self) -> None:
        self.chat_display.clear()
        if self.state:
            self.state.conversation.clear()
            self.state.pending_plan = None
            self.state.pending_tool_calls = None

    def on_load_timeline_info(self) -> None:
        if not self.state:
            self.append_chat("system", "Agent not initialized.")
            return
        tool = TOOL_REGISTRY.get("get_timeline_info")
        if not tool:
            self.append_chat("system", "Tool get_timeline_info not available.")
            return
        result = tool()
        if isinstance(result, dict) and result.get("success") is False:
            self.append_chat("system", f"Timeline info error: {result.get('error')}")
            return
        self.state.timeline_context = result
        self.append_chat("system", f"Timeline loaded: {json.dumps(result)}")

    def on_tool_calls_ready(self, tool_calls: list[dict]) -> None:
        if not self.state:
            return
        results = []
        for call in tool_calls:
            name = call.get("name")
            args = call.get("arguments", {}) or {}
            tool = TOOL_REGISTRY.get(name or "")
            if not tool:
                result = {"success": False, "error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool(**args)
                except Exception as exc:
                    result = {"success": False, "error": str(exc)}
            results.append(result)
            if name == "get_timeline_info" and isinstance(result, dict) and result.get("success", True):
                self.state.timeline_context = result
            self.append_chat("system", f"Tool {name}: {result}")
        self.state.add_message("assistant", "", tool_calls=tool_calls, tool_results=results)
        self.state.pending_tool_calls = None
        self._run_agent("")

    def show_plan_approval_widget(self, plan: EditPlan) -> None:
        self.plan_actions.clear()
        self.plan_summary.setText(edit_plan_summary(plan))
        for action in plan.actions:
            self.plan_actions.addItem(f"{action.action_type}: {action.target}")
        self.plan_frame.setVisible(True)

    def hide_plan_approval_widget(self) -> None:
        self.plan_frame.setVisible(False)

    def append_chat(self, role: str, text: str) -> None:
        if not text:
            return
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if role == "user":
            html = f"<div align='right'><b>You:</b> {safe_text}</div>"
        elif role == "assistant":
            html = f"<div align='left'><b>Agent:</b> {safe_text}</div>"
        else:
            html = f"<div align='left'><i>{safe_text}</i></div>"
        self.chat_display.append(html)

    def _execute_action(self, action) -> str:
        tool_name = action.action_type
        if tool_name == "delete_clip":
            tool_name = "delete_clip_at_playhead"
        elif tool_name == "match_broll":
            tool_name = "match_broll_for_segment"
        elif tool_name == "sync_clips":
            tool_name = "sync_clips_from_timeline"
        tool_func = TOOL_REGISTRY.get(tool_name)
        if not tool_func:
            return "Tool not found"
        try:
            params = dict(action.params or {})
            if action.action_type == "insert_clip":
                params = self._coerce_insert_params(action, params)
            if action.action_type == "delete_clip":
                self._seek_timecode_if_needed(action.target)
            result = tool_func(**params)
            return str(result)
        except Exception as exc:
            return f"Error: {exc}"

    def _format_tool_list(self) -> str:
        lines = []
        for name in TOOL_REGISTRY:
            schema = get_tool_schema(name) or {}
            params = schema.get("input_schema", {}).get("properties", {})
            lines.append(f"{name} ({', '.join(params.keys())})")
        return ", ".join(lines)

    def _coerce_insert_params(self, action, params: dict) -> dict:
        if not params.get("start_frame") and not params.get("start_seconds") and not params.get("start_timecode"):
            target = str(action.target or "").strip()
            if ":" in target:
                params["start_timecode"] = target
            else:
                seconds = self._parse_duration_seconds(target)
                if seconds is not None:
                    params["start_seconds"] = seconds
        target = str(action.target or "").strip().lower()
        if target.startswith("v") and target[1:].isdigit():
            params.setdefault("track_type", "video")
            params.setdefault("track_index", int(target[1:]))
        if target.startswith("a") and target[1:].isdigit():
            params.setdefault("track_type", "audio")
            params.setdefault("track_index", int(target[1:]))
        return params

    @staticmethod
    def _parse_duration_seconds(text: str) -> Optional[float]:
        raw = str(text or "").strip().lower()
        if not raw:
            return None
        if raw.isdigit():
            return float(raw)
        for suffix, scale in (("sec", 1.0), ("secs", 1.0), ("second", 1.0), ("seconds", 1.0), ("s", 1.0)):
            if raw.endswith(suffix):
                value = raw[: -len(suffix)].strip()
                if value.replace(".", "", 1).isdigit():
                    return float(value) * scale
        for suffix, scale in (("min", 60.0), ("mins", 60.0), ("minute", 60.0), ("minutes", 60.0), ("m", 60.0)):
            if raw.endswith(suffix):
                value = raw[: -len(suffix)].strip()
                if value.replace(".", "", 1).isdigit():
                    return float(value) * scale
        return None

    def _seek_timecode_if_needed(self, target: str) -> None:
        if not target:
            return
        target = str(target).strip()
        if ":" not in target:
            return
        try:
            from VideoForge.integrations.resolve_api import ResolveAPI

            api = ResolveAPI()
            timeline = api.get_current_timeline()
            setter = getattr(timeline, "SetCurrentTimecode", None)
            if callable(setter):
                setter(target)
        except Exception:
            return

    def _run_agent(self, text: str) -> None:
        if not self.executor or not self.state:
            self.append_chat("system", "Agent not ready. Set API key in Settings.")
            return
        self.worker = AgentWorker(self.executor, self.state, text)
        self.worker.response_ready.connect(self.on_agent_response)
        self.worker.plan_ready.connect(self.on_plan_ready)
        self.worker.tool_calls_ready.connect(self.on_tool_calls_ready)
        self.worker.start()

