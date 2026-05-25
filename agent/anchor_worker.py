"""Background queue that anchors audit records to Arc without blocking.

The trading loop calls `submit(record)` and returns immediately. A
daemon thread drains the queue, calls Circle's Contract Execution API
(~2-3 seconds per call), and fires a caller-supplied callback with the
resulting tx_hash so the snapshot writer can patch in the arcscan URL.

Failures (Circle 5xx, network blip, etc.) are logged to stderr but do
not crash the worker — the audit record stays in sqlite with
`arc_tx_hash=None` and the trader keeps trading.

The worker also write-aheads each anchored record back to the
`AuditStore` with the tx_hash populated (a *new* row keyed by the same
audit_id; sqlite enforces append-only).
"""

from __future__ import annotations

import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from agent.audit import AuditRecord, AuditStore
from agent.on_chain import AnchorResult, CircleAnchor


# Type alias for the post-anchor callback. Signature:
#   (audit_id, asset_or_none, result_or_none, error_or_none) -> None
OnComplete = Callable[
    [str, str | None, AnchorResult | None, BaseException | None], None
]


class _Pill:
    """Sentinel pushed onto the queue to ask the worker to shut down."""


class AnchorWorker:
    """Non-blocking submit-then-anchor pipeline for audit records.

    Lifecycle:
        worker = AnchorWorker(anchor, store, on_complete=cb)
        worker.submit(record, asset='BTC')   # returns instantly
        ...
        worker.close()                        # drains, joins, shuts down

    The worker thread is daemon so a Ctrl-C on the trading loop kills
    it cleanly without leaking; explicit `close()` is still preferred so
    in-flight anchors finish.
    """

    def __init__(
        self,
        anchor: CircleAnchor,
        store: AuditStore,
        on_complete: OnComplete | None = None,
        max_queue: int = 200,
    ) -> None:
        self._anchor = anchor
        self._store = store
        self._on_complete = on_complete
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="anchor-worker"
        )
        self._thread.start()

    def submit(self, record: AuditRecord, *, asset: str | None = None) -> None:
        """Enqueue one record for anchoring. Non-blocking on a non-full queue.

        Args:
            record: the audit record to anchor. Already persisted via
                `AuditStore.append` before submission (write-ahead invariant
                per `audit-record-and-erc8183-anchor.md` §3).
            asset: convenience tag so the on-complete callback can route
                the tx_hash back to the right open position in the
                snapshot. `None` for non-position-tied anchors.
        """
        try:
            self._queue.put_nowait((record, asset))
        except queue.Full:
            print(
                f"[anchor worker] queue full ({self._queue.maxsize}); "
                f"dropping anchor for audit_id={record.audit_id}",
                file=sys.stderr,
            )

    def qsize(self) -> int:
        return self._queue.qsize()

    def close(self, drain_timeout_s: float = 30.0) -> None:
        """Stop the worker; wait up to `drain_timeout_s` for pending items."""
        try:
            self._queue.put_nowait(_Pill())
        except queue.Full:
            self._stop.set()
        self._thread.join(timeout=drain_timeout_s)

    # ------------------------------------------------------------ internals

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(item, _Pill):
                self._stop.set()
                break
            record, asset = item
            self._anchor_one(record, asset)

    def _anchor_one(self, record: AuditRecord, asset: str | None) -> None:
        result: AnchorResult | None = None
        err: BaseException | None = None
        try:
            result = self._anchor.anchor_record(record)
            # Append a new row with `arc_tx_hash` populated for audit trail.
            updated = replace(
                record,
                arc_tx_hash=result.arc_tx_hash,
                arc_job_id=result.arc_job_id,
            )
            self._store.append(updated)
        except BaseException as e:  # noqa: BLE001
            err = e
            print(
                f"[anchor worker] anchor failed for audit_id="
                f"{record.audit_id}: {e}",
                file=sys.stderr,
            )

        if self._on_complete is not None:
            try:
                self._on_complete(record.audit_id, asset, result, err)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[anchor worker] on_complete callback raised: {e}",
                    file=sys.stderr,
                )
