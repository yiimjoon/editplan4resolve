from typing import Any, Dict


class BrollAdapter:
    """Placeholder for future AutoResolve B-roll integration."""

    def __init__(self) -> None:
        self.enabled = False

    def match(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "stub", "payload": payload}
