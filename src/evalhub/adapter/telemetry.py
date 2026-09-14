"""OpenTelemetry instrumentation for the adapter framework (spans + logs).

Provides ``EvalTracer``, which exposes context-manager methods for the five
evaluation span types defined by the EvalHub observability contract:

- ``evalhub.evaluation.run`` (root)
- ``evalhub.evaluation.dataset_load``
- ``evalhub.evaluation.inference``
- ``evalhub.evaluation.scoring``
- ``evalhub.evaluation.result_log``

:func:`configure_telemetry` also installs an **OTEL Logs pipeline** that
bridges Python ``logging`` to the collector.  Every log record automatically
includes:

- ``severity_text`` / ``severity_number`` mapped from the Python log level
- ``trace_id`` / ``span_id`` from the current span context (log–trace
  correlation)
- ``service.name`` and other ``Resource`` attributes shared with the span
  exporter
- Job-level attributes (``evalhub.job_id``, ``evalhub.benchmark_id``, …) set
  via :func:`set_log_job_context`

When no ``TracerProvider`` is configured (i.e. OTEL environment variables are
absent), the underlying ``trace.get_tracer()`` returns a no-op tracer and all
context managers become zero-cost no-ops.  The log pipeline is likewise
skipped.

Use :func:`configure_telemetry` to install providers before creating any
``EvalTracer`` instances::

    from evalhub.adapter.telemetry import configure_telemetry

    configure_telemetry()  # reads OTEL_* env vars; no-op when unconfigured
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.propagate import extract

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk.trace import TracerProvider

    from .models.job import JobSpec

logger = logging.getLogger(__name__)

_TRACER_NAME = "evalhub.adapter"
_DEFAULT_SERVICE_NAME = "evalhub-adapter"
_lock = threading.Lock()
_provider_installed: TracerProvider | None = None
_owns_provider: bool = False
_log_provider_installed: LoggerProvider | None = None
_owns_log_provider: bool = False

# ---------------------------------------------------------------------------
# Job-level log context
# ---------------------------------------------------------------------------

_job_log_context: ContextVar[dict[str, str]] = ContextVar(
    "evalhub_job_log_context", default={}
)


def set_log_job_context(
    *,
    job_id: str | None = None,
    benchmark_id: str | None = None,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> None:
    """Set job-level attributes injected into every subsequent OTEL log record.

    Call once at the start of a job — typically from
    ``EvalTracer.from_job_spec`` or an adapter's ``main()`` — so that all log
    lines emitted during the run carry the job identity for filtering in the
    observability backend.
    """
    ctx: dict[str, str] = {}
    if job_id is not None:
        ctx["evalhub.job_id"] = job_id
    if benchmark_id is not None:
        ctx["evalhub.benchmark_id"] = benchmark_id
    if provider_id is not None:
        ctx["evalhub.provider_id"] = provider_id
    if model_id is not None:
        ctx["evalhub.model_id"] = model_id
    _job_log_context.set(ctx)


def clear_log_job_context() -> None:
    """Remove all job-level log attributes (e.g. between successive jobs)."""
    _job_log_context.set({})


class _JobContextFilter(logging.Filter):
    """Logging filter that enriches records with job-level OTEL attributes.

    Attributes stored via :func:`set_log_job_context` are copied onto the
    Python ``LogRecord`` so that the OTEL ``LoggingHandler`` forwards them as
    structured log-record attributes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in _job_log_context.get({}).items():
            setattr(record, key, value)
        return True


def configure_telemetry(
    service_name: str | None = None,
    *,
    endpoint: str | None = None,
) -> bool:
    """Install OTLP span and log exporters if OTEL is configured.

    Call this once at adapter startup — before any ``EvalTracer`` is created —
    to enable span and log export.  The function is **thread-safe** and
    **idempotent**: concurrent or repeated calls are serialised by a lock and
    return ``True`` immediately once a provider has been accepted.

    If a ``TracerProvider`` was already installed globally (e.g. by OTEL
    auto-instrumentation or a test harness), it is reused and no new provider
    is created.

    Configuration is resolved from arguments first, then from standard OTEL
    environment variables:

    * ``OTEL_EXPORTER_OTLP_ENDPOINT`` — collector endpoint
      (e.g. ``http://localhost:4317``)
    * ``OTEL_SERVICE_NAME`` — logical service name
      (defaults to ``evalhub-adapter``)

    When neither ``endpoint`` nor ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set the
    function returns ``False`` and no provider is installed, preserving the
    default no-op tracer behaviour.

    **Logs pipeline** — when OTEL is configured, a ``LoggerProvider`` with
    ``OTLPLogExporter`` is installed and a ``LoggingHandler`` is added to the
    Python root logger.  Every Python log record is exported as a structured
    OTEL log record with ``severity_text``, ``severity_number``,
    ``trace_id``/``span_id`` (from the active span), and any job-level
    attributes set via :func:`set_log_job_context`.

    Args:
        service_name: Override for ``OTEL_SERVICE_NAME``.
        endpoint: Override for ``OTEL_EXPORTER_OTLP_ENDPOINT``.

    Returns:
        ``True`` if a ``TracerProvider`` is active (installed by us, reused
        from an existing global, or already accepted by a previous call),
        ``False`` if OTEL is not configured.
    """
    global _provider_installed, _owns_provider  # noqa: PLW0603

    if _provider_installed is not None:
        return True

    with _lock:
        if _provider_installed is not None:
            return True

        resolved_endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not resolved_endpoint:
            logger.debug(
                "OTEL_EXPORTER_OTLP_ENDPOINT not set — skipping telemetry setup"
            )
            return False

        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider as _TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            logger.warning(
                "opentelemetry-exporter-otlp-proto-grpc is not installed. "
                "Install it with: pip install opentelemetry-exporter-otlp-proto-grpc"
            )
            return False

        # If an SDK TracerProvider is already globally active (e.g. from OTEL
        # auto-instrumentation or a test harness), reuse it rather than
        # creating a second provider that set_tracer_provider would discard.
        current = trace.get_tracer_provider()
        if isinstance(current, _TracerProvider):
            _provider_installed = current
            _owns_provider = False
            logger.info(
                "Reusing existing TracerProvider (service=%s)",
                dict(current.resource.attributes).get("service.name", "unknown"),
            )
            return True

        resolved_name = (
            service_name or os.environ.get("OTEL_SERVICE_NAME") or _DEFAULT_SERVICE_NAME
        )

        resource = Resource.create({"service.name": resolved_name})
        exporter = OTLPSpanExporter(endpoint=resolved_endpoint)
        provider = _TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        # set_tracer_provider uses a set-once guard.  If another thread won
        # the race between our isinstance check above and the set call, our
        # provider was silently discarded.  Detect that and clean up.
        active = trace.get_tracer_provider()
        if active is not provider:
            provider.shutdown()
            if isinstance(active, _TracerProvider):
                _provider_installed = active
                _owns_provider = False
                logger.info(
                    "Another TracerProvider was installed concurrently — "
                    "reusing it (service=%s)",
                    dict(active.resource.attributes).get("service.name", "unknown"),
                )
                return True
            logger.warning(
                "set_tracer_provider was ignored and no SDK provider is active. "
                "Spans will be emitted via the no-op tracer."
            )
            return False

        _provider_installed = provider
        _owns_provider = True
        atexit.register(_shutdown_provider)

        logger.info(
            "TracerProvider installed (service=%s, endpoint=%s)",
            resolved_name,
            resolved_endpoint,
        )

        _configure_log_pipeline(resource, resolved_endpoint)

        return True


def _configure_log_pipeline(resource: Any, endpoint: str) -> None:
    """Install a ``LoggerProvider`` + OTLP exporter and bridge Python logging.

    Called internally by :func:`configure_telemetry` after the span pipeline is
    set up.  Adds a ``LoggingHandler`` to the root Python logger so that every
    ``logging.info(…)`` call is exported as a structured OTEL log record with
    correct ``severity_text``, ``severity_number``, ``trace_id``/``span_id``,
    resource attributes, and job-level attributes.
    """
    global _log_provider_installed, _owns_log_provider  # noqa: PLW0603

    try:
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider as _LoggerProvider
        from opentelemetry.sdk._logs import LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    except ImportError:
        logger.debug("OTEL log exporter not available — log pipeline skipped")
        return

    log_exporter = OTLPLogExporter(endpoint=endpoint)
    log_provider = _LoggerProvider(resource=resource)
    log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))

    handler = LoggingHandler(level=logging.NOTSET, logger_provider=log_provider)
    handler.addFilter(_JobContextFilter())
    logging.getLogger().addHandler(handler)

    _log_provider_installed = log_provider
    _owns_log_provider = True

    logger.info("OTEL LoggerProvider installed (endpoint=%s)", endpoint)


def _shutdown_provider() -> None:
    """Flush and shut down providers registered by :func:`configure_telemetry`.

    Only shuts down providers if we created them (not when we reused an
    existing global provider).
    """
    global _provider_installed, _owns_provider  # noqa: PLW0603
    global _log_provider_installed, _owns_log_provider  # noqa: PLW0603
    with _lock:
        if _log_provider_installed is not None and _owns_log_provider:
            try:
                _log_provider_installed.shutdown()
            except Exception:
                logger.debug("LoggerProvider shutdown error", exc_info=True)
        _log_provider_installed = None
        _owns_log_provider = False

        if _provider_installed is not None and _owns_provider:
            try:
                _provider_installed.shutdown()
            except Exception:
                logger.debug("TracerProvider shutdown error", exc_info=True)
        _provider_installed = None
        _owns_provider = False


class _EnvCarrier(dict[str, str]):
    """Read-only carrier that extracts W3C TraceContext headers from env vars.

    Implements ``Mapping[str, str]`` so the default OTEL ``DefaultGetter``
    can call ``.get(key)`` on it.  The EvalHub sidecar (or Kubernetes
    downward API) injects ``TRACEPARENT`` and optionally ``TRACESTATE``
    into the pod environment.
    """

    def __init__(self) -> None:
        data: dict[str, str] = {}
        for header in ("traceparent", "tracestate"):
            val = os.environ.get(header.upper())
            if val:
                data[header] = val
        super().__init__(data)


class EvalTracer:
    """Thin wrapper around an OTEL ``Tracer`` scoped to one evaluation job.

    This class is an **instrumentation library** and intentionally does *not*
    create or configure a ``TracerProvider``.  The host application (or the
    Kubernetes sidecar / OTEL auto-configuration) must set up a
    ``TracerProvider`` with the desired ``Resource``, exporter, and span
    processor *before* any spans are created.  When no provider is configured,
    ``trace.get_tracer()`` returns a no-op tracer and all context managers
    become zero-cost no-ops.

    Typical usage via ``DefaultCallbacks``::

        callbacks = DefaultCallbacks.from_adapter(adapter)

        with callbacks.tracer.evaluation_run():
            with callbacks.tracer.dataset_load(source="hf://mmlu", size_samples=500):
                ...
            for batch in batches:
                with callbacks.tracer.inference_batch(
                    model_id="llama-3", batch_size=32
                ):
                    ...
            with callbacks.tracer.scoring(benchmark="mmlu", scorer="accuracy"):
                ...
            with callbacks.tracer.result_log():
                callbacks.report_results(results)
    """

    def __init__(self) -> None:
        self._tracer = trace.get_tracer(_TRACER_NAME, schema_url=None)
        self._job_id: str | None = None
        self._provider: str | None = None
        self._collection: str | None = None
        self._model_id: str | None = None

    @classmethod
    def from_job_spec(cls, job_spec: JobSpec) -> EvalTracer:
        """Create a tracer pre-populated with job identity attributes.

        Also calls :func:`set_log_job_context` so that all subsequent Python
        log records carry the same job identity as OTEL log-record attributes.
        """
        tracer = cls()
        tracer._job_id = job_spec.id
        tracer._provider = job_spec.provider_id
        tracer._collection = job_spec.benchmark_id
        tracer._model_id = job_spec.model.name if job_spec.model else None

        set_log_job_context(
            job_id=job_spec.id,
            benchmark_id=job_spec.benchmark_id,
            provider_id=job_spec.provider_id,
            model_id=job_spec.model.name if job_spec.model else None,
        )
        return tracer

    @staticmethod
    def extract_context() -> otel_context.Context | None:
        """Extract W3C TraceContext from environment variables.

        Returns a ``Context`` carrying the remote span context injected by
        the EvalHub sidecar/server, or ``None`` if no traceparent is present.
        """
        ctx = extract(carrier=_EnvCarrier())
        span = trace.get_current_span(ctx)
        if span.get_span_context().trace_id == 0:
            return None
        return ctx

    @contextmanager
    def evaluation_run(self) -> Generator[trace.Span, None, None]:
        """Root span wrapping the full evaluation execution."""
        parent_ctx = self.extract_context()
        attrs: dict[str, str] = {}
        if self._job_id is not None:
            attrs["evalhub.job_id"] = self._job_id
        if self._provider is not None:
            attrs["evalhub.provider"] = self._provider
        if self._collection is not None:
            attrs["evalhub.collection"] = self._collection
        if self._model_id is not None:
            attrs["evalhub.model_id"] = self._model_id

        ctx = parent_ctx if parent_ctx is not None else otel_context.get_current()
        with self._tracer.start_as_current_span(
            "evalhub.evaluation.run",
            context=ctx,
            attributes=attrs,
        ) as span:
            yield span

    @contextmanager
    def dataset_load(
        self,
        source: str | None = None,
        size_samples: int | None = None,
    ) -> Generator[trace.Span, None, None]:
        """Child span for the dataset loading phase."""
        attrs: dict[str, str | int] = {}
        if source is not None:
            attrs["dataset.source"] = source
        if size_samples is not None:
            attrs["dataset.size_samples"] = size_samples

        with self._tracer.start_as_current_span(
            "evalhub.evaluation.dataset_load",
            attributes=attrs,
        ) as span:
            yield span

    @contextmanager
    def inference_batch(
        self,
        model_id: str | None = None,
        batch_size: int | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
    ) -> Generator[trace.Span, None, None]:
        """Child span for a single inference batch."""
        attrs: dict[str, str | int] = {}
        if model_id is not None:
            attrs["model.id"] = model_id
        if batch_size is not None:
            attrs["batch.size"] = batch_size
        if tokens_input is not None:
            attrs["tokens.input"] = tokens_input
        if tokens_output is not None:
            attrs["tokens.output"] = tokens_output

        with self._tracer.start_as_current_span(
            "evalhub.evaluation.inference",
            attributes=attrs,
        ) as span:
            yield span

    @contextmanager
    def scoring(
        self,
        benchmark: str | None = None,
        scorer: str | None = None,
    ) -> Generator[trace.Span, None, None]:
        """Child span for the scoring / post-processing phase."""
        attrs: dict[str, str] = {}
        if benchmark is not None:
            attrs["benchmark.name"] = benchmark
        if scorer is not None:
            attrs["scorer.type"] = scorer

        with self._tracer.start_as_current_span(
            "evalhub.evaluation.scoring",
            attributes=attrs,
        ) as span:
            yield span

    @contextmanager
    def result_log(self) -> Generator[trace.Span, None, None]:
        """Child span for result logging to MLflow and EvalHub service."""
        with self._tracer.start_as_current_span(
            "evalhub.evaluation.result_log",
        ) as span:
            yield span
