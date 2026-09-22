import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Set

logger = logging.getLogger(__name__)


class WorkspaceEventBus:
    """
    In-memory Server-Sent Events (SSE) workspace event bus providing real-time
    synchronization across enterprise workspace collaborators. Broadcasts document
    ingestion updates, membership changes, and thread activities with sub-millisecond latency.
    """

    def __init__(self):
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def subscribe(self, workspace_id: str) -> asyncio.Queue:
        """Register an active connection queue for a workspace channel."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        lock = self._get_lock()
        async with lock:
            if workspace_id not in self._subscribers:
                self._subscribers[workspace_id] = set()
            self._subscribers[workspace_id].add(queue)
        return queue

    async def unsubscribe(self, workspace_id: str, queue: asyncio.Queue):
        """Unregister an active connection queue and clean up channel if empty."""
        lock = self._get_lock()
        async with lock:
            if workspace_id in self._subscribers:
                self._subscribers[workspace_id].discard(queue)
                if not self._subscribers[workspace_id]:
                    del self._subscribers[workspace_id]

    async def publish(self, workspace_id: str, event_type: str, data: Any):
        """Broadcast an event payload to all active workspace subscribers."""
        lock = self._get_lock()
        async with lock:
            queues = list(self._subscribers.get(workspace_id, []))
        if not queues:
            return
        payload = {"type": event_type, "workspace_id": workspace_id, "data": data}
        for q in queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                logger.warning("Subscriber queue full for workspace %s, dropping event.", workspace_id)
            except Exception as e:
                logger.debug("Error delivering event to subscriber: %s", e)

    def publish_sync(self, workspace_id: str, event_type: str, data: Any):
        """Thread-safe synchronization helper for synchronous FastAPI routes and background jobs."""
        payload = {"type": event_type, "workspace_id": workspace_id, "data": data}
        loop = self._loop
        try:
            if loop is None or loop.is_closed():
                loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            def _push_all():
                queues = list(self._subscribers.get(workspace_id, []))
                for q in queues:
                    try:
                        q.put_nowait(payload)
                    except Exception:
                        pass
            try:
                loop.call_soon_threadsafe(_push_all)
            except RuntimeError:
                pass
        else:
            queues = list(self._subscribers.get(workspace_id, []))
            for q in queues:
                try:
                    q.put_nowait(payload)
                except Exception:
                    pass

    async def sse_generator(self, workspace_id: str, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """Stream events formatted conforming to the W3C Server-Sent Events standard."""
        # Initial greeting event
        yield f"event: connected\ndata: {json.dumps({'status': 'connected', 'workspace_id': workspace_id})}\n\n"
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: {event['type']}\ndata: {json.dumps(event['data'])}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive heartbeat comment to prevent proxy and browser timeouts
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await self.unsubscribe(workspace_id, queue)


# Global singleton instance
workspace_event_bus = WorkspaceEventBus()
