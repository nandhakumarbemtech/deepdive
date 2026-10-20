"""Tiny in-process counters exposed in Prometheus text format at /metrics (no extra dependency)."""
import threading

_lock = threading.Lock()
_counters = {}  # (name, sorted label tuple) -> value


def inc(name, value=1.0, **labels):
    """Increase a counter; labels (e.g. model="gpt-4o") create separate series."""
    key = (name, tuple(sorted(labels.items())))
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + value


def snapshot():
    """Return a copy of all counters (used by tests)."""
    with _lock:
        return dict(_counters)


def render():
    """Render all counters in the Prometheus exposition format."""
    lines = []
    with _lock:
        for (name, labels), value in sorted(_counters.items()):
            lab = ",".join(f'{k}="{v}"' for k, v in labels)
            lines.append(f"deepdive_{name}{{{lab}}} {value}" if lab else f"deepdive_{name} {value}")
    return "\n".join(lines) + "\n"
