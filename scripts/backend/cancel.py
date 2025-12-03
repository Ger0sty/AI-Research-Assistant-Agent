# scripts/backend/cancel.py
from threading import Event, Lock
from contextlib import contextmanager
from typing import Dict

class CancelRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._tokens: Dict[str, Event] = {}

    def create_token(self, search_id: str) -> Event:
        """Create & register a new cancellation token for this search_id."""
        evt = Event()
        with self._lock:
            self._tokens[search_id] = evt
        return evt

    def get_token(self, search_id: str) -> Event | None:
        with self._lock:
            return self._tokens.get(search_id)

    def cancel(self, search_id: str) -> bool:
        """Mark the search as cancelled. Returns True iff it existed."""
        with self._lock:
            evt = self._tokens.get(search_id)
            if not evt:
                return False
            evt.set()
            return True

    def clear(self, search_id: str) -> None:
        """Remove the token once the search finishes / errors."""
        with self._lock:
            self._tokens.pop(search_id, None)

    @contextmanager
    def token_for(self, search_id: str):
        """Context manager that auto-clears the token when done."""
        evt = self.create_token(search_id)
        try:
            yield evt
        finally:
            self.clear(search_id)


cancel_registry = CancelRegistry()
