from __future__ import annotations

import os
from typing import Optional

from opentelemetry import trace


def init_tracing(service_name: str = "scopebench", enable_console: Optional[bool] = None) -> None:
    """Initialize OpenTelemetry tracing.

    This function is *optional* at runtime.

    - If `opentelemetry-sdk` is installed, it configures a TracerProvider.
    - If not installed, tracing becomes a no-op (but the rest of ScopeBench works).

    MVP defaults to ConsoleSpanExporter if:
    - enable_console is True, or
    - SCOPEBENCH_OTEL_CONSOLE=1
    """
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    except Exception:
        # No SDK available; leave default no-op provider.
        return

    if enable_console is None:
        enable_console = os.getenv("SCOPEBENCH_OTEL_CONSOLE", "0") in {"1", "true", "True"}

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if enable_console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def get_tracer(name: str = "scopebench"):
    return trace.get_tracer(name)
