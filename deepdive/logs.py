"""Structured JSON logging so logs can be searched in any log platform (Datadog, Loki, CloudWatch)."""
import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    """Format each record as one JSON object per line, including any `extra=` fields."""

    RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

    def format(self, record):
        out = {"ts": round(time.time(), 3), "level": record.levelname,
               "logger": record.name, "msg": record.getMessage()}
        out.update({k: v for k, v in record.__dict__.items() if k not in self.RESERVED})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup(level="INFO"):
    """Install the JSON formatter on the root logger once."""
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root.setLevel(level)
