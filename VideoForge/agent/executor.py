"""Agent execution loop for VideoForge (Gemini-first)."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional
from urllib import request
from urllib.error import HTTPError, URLError

from VideoForge.agent.guardrails import validate_plan_safety
from VideoForge.agent.planner import EditPlan, edit_plan_from_dict, edit_plan_summary
from VideoForge.agent.runtime import AgentState
from VideoForge.agent.tools import TOOL_REGISTRY, get_tool_schema
from VideoForge.ai.llm_hooks import get_llm_api_key, get_llm_model, is_llm_enabled
from VideoForge.config.config_manager import Config

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are VideoForge Agent, an AI assistant for professional video editing automation in DaVinci Resolve within the VideoForge ecosystem. Your goal is to make editing faster, safer, and smarter while preserving professional quality.

Scope
You can help with:
- Timeline read/analysis (timeline info, playhead, clips-at-playhead).
- Planning and executing edits: insert, delete, move, cut, sync.
- B-roll matching via library search (FTS + vector/CLIP) and transcript-driven queries.
- Audio sync via silence/voice/content analysis.
- Optional AI B-roll generation via ComfyUI (when configured).

Hard Constraints (must follow)
- Resolve API calls must run on the main thread only. If a request would run from a worker/background thread, do not execute; explain and instruct user to use the app UI flow (e.g., load timeline info properly).
- Default to non-destructive Draft Mode behavior: do not assume destructive edits are final without explicit approval.
- Do not unload/delete Whisper models (unloading can crash). Keep models resident once loaded.
- For batch operations, limit autonomous execution; split into smaller approved chunks when large.

Operating Modes
- Recommend Only (read-only): Produce an EditPlan JSON only; do not execute modification actions.
- Approve Required (default): You may run read-only queries immediately. For modifications, always generate EditPlan, present summary/steps/risk/warnings, wait for approval, then execute.
- Full Access (advanced): Execute without explicit approval, but cap automatic actions to agent_max_auto_actions (default 5). Still warn before destructive actions and stop if risk is high/unclear.

Execution Rules
- Always start by ensuring timeline context is available; if missing/stale, request/trigger timeline info first.
- Interpret intent: "here/current position" -> playhead; explicit timecode -> HH:MM:SS:FF; "track 2" -> track index mapping (V1.., A1..).
- For insert_clip, include start_timecode or start_seconds in params when the user specifies time.
- Before delete: warn, state what will be removed, require confirmation in Approve Required mode.
- Before sync: confirm clips have audio; if low confidence/low audio, warn and suggest normalization or switching to voice mode.
 - Only include actions required for the most recent user request; ignore prior requests.

Error Handling
When a tool/action fails, report the real reason and give a next step:
- "Timeline not found" -> user must open a timeline in Resolve, then load timeline info.
- "Main thread required" -> instruct user to use the correct UI/flow that runs on main thread.
- "Clip not found" -> ask for exact clip name/track/timecode; offer to inspect clips near playhead.

EditPlan JSON Schema
When producing a plan, output JSON only in this format:
{
  "request": "verbatim user request",
  "summary": "1-2 sentences",
  "actions": [
    {
      "action_type": "insert_clip|delete_clip|move_clip|cut_clip|sync_clips|match_broll|apply_transition",
      "target": "V1|A1|timecode|clip_name",
      "params": {},
      "reason": "user-facing explanation"
    }
  ],
  "estimated_clips": 0,
  "risk_level": "safe|moderate|destructive",
  "warnings": []
}

Communication Style
Professional, concise, no fluff.
Ask clarifying questions when target/track/timecode/duration is ambiguous.
Prefer plan-first when uncertain.
"""


class AgentExecutor:
    """LLM tool-calling loop with JSON tool calls."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        if not is_llm_enabled():
            raise ValueError("LLM provider disabled. Enable LLM in Settings to use the agent.")
        agent_key = str(Config.get("agent_api_key", "")).strip()
        llm_key = get_llm_api_key()
        self.api_key = api_key or agent_key or (llm_key or "")
        if not self.api_key:
            raise ValueError("Agent API key missing. Set Config agent_api_key or llm_api_key.")
        agent_model = str(Config.get("agent_model", "")).strip()
        llm_model = str(get_llm_model() or "").strip()
        if agent_model in {"", "gemini-2.0-flash-exp"} and llm_model:
            agent_model = llm_model
        self.model = model or agent_model
        if not self.model:
            self.model = "gemini-2.0-flash-exp"
        max_tokens = Config.get("agent_max_tokens", 0) or Config.get("llm_max_tokens", 4096)
        try:
            self.max_tokens = int(max_tokens)
        except Exception:
            self.max_tokens = 4096
        self.max_steps = 6

    def run(self, state: AgentState, user_input: str) -> str:
        if user_input:
            state.add_message("user", user_input)

        for _ in range(self.max_steps):
            prompt = self._build_prompt(state)
            response_text = self._call_gemini(prompt)
            parsed = self._parse_agent_response(response_text)

            if parsed.get("type") == "tool_calls":
                tool_calls = parsed.get("tool_calls", [])
                if not tool_calls:
                    return "Tool call requested but no tools were provided."
                if state.mode == "recommend_only":
                    return "Tool execution is disabled in recommend_only mode. Return an EditPlan instead."
                if state.mode == "approve_required":
                    blocked = [call for call in tool_calls if call.get("name") not in self._safe_tools()]
                    if blocked:
                        return "Tool execution is blocked in approve_required mode. Return an EditPlan instead."
                state.pending_tool_calls = tool_calls
                return ""

            plan_data = parsed.get("edit_plan")
            if plan_data:
                try:
                    plan = edit_plan_from_dict(plan_data)
                except Exception as exc:
                    return f"Plan parse error: {exc}. Please output a complete EditPlan JSON."
                warnings = validate_plan_safety(plan)
                for warning in warnings:
                    if warning not in plan.warnings:
                        plan.warnings.append(warning)
                state.pending_plan = plan
                summary = edit_plan_summary(plan)
                if state.mode == "full_access":
                    return f"{summary} Executing plan now."
                return f"{summary} Please approve or modify the plan."

            final_text = parsed.get("response")
            if not final_text:
                final_text = response_text
            state.add_message("assistant", final_text)
            return final_text

        return "Max agent steps reached without final response."

    def _build_prompt(self, state: AgentState) -> str:
        timeline_context = (
            json.dumps(state.timeline_context, indent=2)
            if state.timeline_context
            else "Not loaded. Use get_timeline_info first."
        )
        conversation = self._format_conversation(state)
        return (
            DEFAULT_SYSTEM_PROMPT.strip()
            + "\n\n"
            + f"Current Mode: {state.mode}\n"
            + "- recommend_only: Generate EditPlan JSON only. Do NOT execute tools.\n"
            + "- approve_required: Generate EditPlan and wait for approval.\n"
            + f"- full_access: Execute tools immediately (max {Config.get('agent_max_auto_actions', 5)} actions).\n\n"
            + "Available Tools:\n"
            + f"{self._format_tool_list()}\n\n"
            + "Timeline Context:\n"
            + f"{timeline_context}\n\n"
            + "Mode rule: In recommend_only, do NOT call tools; output EditPlan only.\n\n"
            + "Return JSON only. Valid formats:\n"
            + '{"type":"tool_calls","tool_calls":[{"name":"tool","arguments":{}}]}\n'
            + '{"type":"final","response":"text response"}\n'
            + '{"type":"plan","edit_plan":{...},"response":"optional summary"}\n\n'
            + "EditPlan required keys:\n"
            + '{"request":"","summary":"","actions":[{"action_type":"","target":"","params":{},"reason":""}],'
            + '"estimated_clips":0,"risk_level":"safe","warnings":[]}\n\n'
            + f"Conversation:\n{conversation}\n\n"
            + "Your JSON response:\n"
        )

    def _format_conversation(self, state: AgentState) -> str:
        lines: List[str] = []
        for msg in state.conversation:
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                if msg.content:
                    lines.append(f"Assistant: {msg.content}")
            if msg.tool_results:
                for result in msg.tool_results:
                    lines.append(f"ToolResult: {json.dumps(result, ensure_ascii=False)}")
        return "\n".join(lines) if lines else "User: (no messages yet)"

    def _format_tool_list(self) -> str:
        lines = []
        for name in TOOL_REGISTRY:
            schema = get_tool_schema(name) or {}
            description = schema.get("description", "")
            params = schema.get("input_schema", {}).get("properties", {})
            params_list = ", ".join(sorted(params.keys())) or "no parameters"
            lines.append(f"- {name}: {description} (params: {params_list})")
        return "\n".join(lines)

    def _parse_agent_response(self, text: str) -> Dict[str, Any]:
        try:
            data = self._extract_json(text)
        except Exception:
            return {"type": "final", "response": text}
        if not isinstance(data, dict):
            return {"type": "final", "response": text}
        if self._looks_like_plan(data):
            return {"type": "plan", "edit_plan": data}
        response_type = data.get("type")
        if response_type == "tool_calls":
            calls = data.get("tool_calls") or data.get("calls") or []
            return {"type": "tool_calls", "tool_calls": calls}
        if response_type == "plan":
            return {
                "type": "plan",
                "edit_plan": data.get("edit_plan") or data.get("plan"),
                "response": data.get("response"),
            }
        if response_type == "final":
            return {"type": "final", "response": data.get("response", "")}
        if data.get("edit_plan"):
            return {"type": "plan", "edit_plan": data.get("edit_plan")}
        return {"type": "final", "response": text}

    def _execute_tools(self, tool_calls: List[Dict[str, Any]], state: AgentState) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        safe_tools = {"get_timeline_info", "get_clips_at_playhead", "match_broll_for_segment"}
        for call in tool_calls:
            name = call.get("name")
            arguments = call.get("arguments", {}) or {}
            if not name:
                results.append({"success": False, "error": "Tool name missing."})
                continue
            if state.mode == "recommend_only":
                results.append(
                    {
                        "success": False,
                        "error": "Tool execution disabled in recommend_only mode.",
                        "suggestion": "Generate an EditPlan instead.",
                    }
                )
                continue
            if state.mode == "approve_required" and name not in safe_tools:
                results.append(
                    {
                        "success": False,
                        "error": "Tool execution blocked in approve_required mode.",
                        "suggestion": "Generate an EditPlan for approval instead.",
                    }
                )
                continue
            result = self._execute_tool(name, arguments, state)
            results.append(result)
        return results

    def _execute_tool(self, name: str, params: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        func = TOOL_REGISTRY.get(name)
        if func is None:
            return {"success": False, "error": f"Unknown tool: {name}"}
        try:
            result = func(**params)
        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "suggestion": "Check timeline availability or inputs.",
            }
        if name == "get_timeline_info" and isinstance(result, dict) and result.get("success", True):
            state.timeline_context = result
        if isinstance(result, dict):
            return result
        return {"success": True, "result": result}

    def _execute_plan_immediately(self, plan: EditPlan) -> str:
        max_actions = int(Config.get("agent_max_auto_actions", 5))
        if len(plan.actions) > max_actions:
            return f"Plan has {len(plan.actions)} actions. Limit is {max_actions}."
        results = []
        for action in plan.actions:
            tool_name = action.action_type
            if tool_name == "delete_clip":
                tool_name = "delete_clip_at_playhead"
            if tool_name == "match_broll":
                tool_name = "match_broll_for_segment"
            if tool_name == "sync_clips":
                tool_name = "sync_clips_from_timeline"
            func = TOOL_REGISTRY.get(tool_name)
            if func is None:
                results.append(f"{action.action_type}: tool not found")
                continue
            try:
                result = func(**action.params)
                results.append(f"{action.action_type}: {result}")
            except Exception as exc:
                results.append(f"{action.action_type}: error {exc}")
        return "\n".join(results) if results else "Plan executed."

    def _call_gemini(self, prompt: str) -> str:
        model = self.model
        if model.startswith("models/"):
            model = model[len("models/") :]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"Gemini HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Gemini network error: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        try:
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            raise RuntimeError(f"Gemini response parse failed: {exc}") from exc

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        raw = str(text or "")
        start_obj = raw.find("{")
        start_arr = raw.find("[")
        if start_obj == -1 and start_arr == -1:
            raise ValueError("Response did not contain JSON.")
        if start_obj == -1:
            start = start_arr
        elif start_arr == -1:
            start = start_obj
        else:
            start = min(start_obj, start_arr)
        decoder = json.JSONDecoder()
        try:
            parsed, _end = decoder.raw_decode(raw[start:])
            return parsed
        except json.JSONDecodeError:
            end = raw.rfind("}")
            if end == -1 or end <= start:
                raise ValueError("Response did not contain JSON.")
            payload = raw[start : end + 1]
            return json.loads(payload)

    @staticmethod
    def _looks_like_plan(data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        if "actions" not in data:
            return False
        if data.get("type") in {"plan", "edit_plan"}:
            return True
        return any(key in data for key in ("summary", "request", "risk_level"))

    @staticmethod
    def _safe_tools() -> set[str]:
        return {"get_timeline_info", "get_clips_at_playhead", "match_broll_for_segment"}

    @staticmethod
    def test_connection(api_key: str, model: str) -> str:
        executor = AgentExecutor(api_key=api_key, model=model)
        return executor._call_gemini("Return JSON: {\"ok\": true}")

