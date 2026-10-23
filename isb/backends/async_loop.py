"""One persistent asyncio loop for a backend built on vLLM's AsyncLLM (design.md §12.17).

Shared by nnsight's async backend and plain AsyncLLM serving. `asyncio.run` creates and closes a
fresh loop per call, which kills the engine's background output loop and makes the next call raise
EngineDeadError, so a backend keeps one loop for its lifetime. Outside a concurrent session each
call drives the loop on the calling thread; inside one (`concurrent_session`) the loop runs on its
own thread and every client thread submits to it, so overlapping requests reach the engine's
scheduler together: this is what `concurrent_requests` declares.
"""
from __future__ import annotations

import contextlib


class AsyncLoopMixin:
    # Several client threads may submit at once (design.md §12.17): AsyncLLM is a serving engine.
    concurrent_requests = True

    _loop = None
    _loop_thread = None          # set while a concurrent session runs the loop

    def __getstate__(self):
        # A cell's build closure may capture a bound method of the backend, so the backend is
        # serialized into the mediator sent to the worker. The loop is not picklable (it holds a
        # `_contextvars.Context`) and the worker never needs it: drop it, and its thread.
        state = self.__dict__.copy()
        state["_loop"] = None
        state["_loop_thread"] = None
        return state

    def _run_coro(self, coro):
        """Drive `coro` on the persistent loop. Each call runs in a fresh copied contextvars
        Context, as `asyncio.run` does; without the copy nnsight's mediator serialization captures
        a `_contextvars.Context` (`TypeError: cannot pickle`)."""
        import asyncio
        import contextvars

        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        if self._loop_thread is not None:            # a concurrent session: the loop runs there
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        return contextvars.copy_context().run(self._loop.run_until_complete, coro)

    @contextlib.contextmanager
    def concurrent_session(self):
        """Run the persistent loop on its own thread while clients submit from theirs. Each
        call's coroutine is scheduled from the client thread, so it runs in a copy of that
        thread's context (as `run_until_complete` above does). An nnsight trace's body runs
        inside its coroutine up to the `with` exit, which submits it before the first await, so
        two clients' traces never interleave on the model's one interleaver (nnsight's own
        server relies on the same property, `nnsight/modeling/vllm/serve/server.py`)."""
        import asyncio
        import threading

        if self._loop_thread is not None:
            raise RuntimeError("a concurrent session is already open")
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        loop = self._loop
        thread = threading.Thread(target=loop.run_forever, name="isb-vllm-loop", daemon=True)
        thread.start()
        self._loop_thread = thread
        try:
            yield
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            self._loop_thread = None

    def _close_loop(self):
        """Close the loop after the engine shut down: cancel the engine's handlers and let their
        finally blocks run first (otherwise vLLM reports "Event loop is closed" at exit)."""
        import asyncio

        if self._loop is None or self._loop.is_closed():
            self._loop = None
            return

        async def drain():
            pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await self._loop.shutdown_asyncgens()

        with contextlib.suppress(Exception):
            self._loop.run_until_complete(drain())
        with contextlib.suppress(Exception):
            self._loop.close()
        self._loop = None
