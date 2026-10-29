"""Rank-local read tasks share persistent submission and cleanup workers."""

from __future__ import annotations

from collections import deque
import threading
import time


class ReadJob:
    def __init__(self, task):
        self.task = task
        self.done = threading.Event()
        self.error = None

    def join(self, timeout=None):
        self.done.wait(timeout)

    def is_alive(self):
        return not self.done.is_set()


class ReadService:
    def __init__(self, store):
        self._store = store
        self._condition = threading.Condition()
        self._tasks = (deque(), deque())
        self._cleanup = deque()
        self._background_owner = None
        self._background_waiters = deque()
        self._closing = False
        self._wake_generation = 0
        self._thread = threading.Thread(
            target=self._run, name="tutti-read-feeder", daemon=True)
        self._cleanup_thread = threading.Thread(
            target=self._run_cleanup, name="tutti-read-cleanup", daemon=True)
        self._thread.start()
        self._cleanup_thread.start()

    def submit(self, task, priority=0):
        job = ReadJob(task)
        with self._condition:
            if self._closing:
                raise RuntimeError("read service is closing")
            self._tasks[priority].append(job)
            self._wake_generation += 1
            self._condition.notify_all()
        return job

    def submit_cleanup(self, callback):
        job = ReadJob(callback)
        with self._condition:
            if self._closing:
                raise RuntimeError("read service is closing")
            self._cleanup.append(job)
            self._condition.notify_all()
        return job

    def wake(self):
        with self._condition:
            self._wake_generation += 1
            self._condition.notify_all()

    def acquire_background(self, task):
        with self._condition:
            if self._background_owner is task:
                return True
            if task not in self._background_waiters:
                self._background_waiters.append(task)
            if self._background_owner is None and self._background_waiters[0] is task:
                self._background_owner = self._background_waiters.popleft()
                return True
            return False

    def release_background(self, task):
        with self._condition:
            if self._background_owner is task:
                self._background_owner = None
            if task in self._background_waiters:
                self._background_waiters.remove(task)
            self._wake_generation += 1
            self._condition.notify_all()

    def join_events(self, events):
        import torch

        event_type = torch.cuda.Event
        if not isinstance(event_type, type) or not all(
                isinstance(event, event_type) for event in events):
            return None
        stream = getattr(self, "_foreground_gate_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self._store._accel_id)
            self._foreground_gate_stream = stream
        for event in events:
            stream.wait_event(event)
        done = event_type()
        done.record(stream)
        return done

    def _bind_device(self):
        from tutti.engine.core import _bind_thread_cuda_device
        _bind_thread_cuda_device(self._store)

    def _needs_poll(self, job):
        try:
            return bool(getattr(job.task, "needs_poll", lambda: True)())
        except Exception as exc:
            job.error = exc
            try:
                job.task.fail(exc)
            except Exception as failure:
                job.error = failure
            return True

    def _run(self):
        self._bind_device()
        while True:
            with self._condition:
                while not any(self._tasks):
                    if self._closing:
                        return
                    self._condition.wait()
                jobs = tuple(self._tasks[0]) + tuple(self._tasks[1])
                generation = self._wake_generation
            for job in jobs:
                try:
                    finished = bool(job.task.advance())
                except Exception as exc:
                    job.error = exc
                    try:
                        job.task.fail(exc)
                    except Exception as failure:
                        job.error = failure
                    finished = True
                if finished:
                    with self._condition:
                        for pending in self._tasks:
                            if job in pending:
                                pending.remove(job)
                                break
                        job.done.set()
                        self._wake_generation += 1
                        self._condition.notify_all()
            with self._condition:
                pending = tuple(self._tasks[0]) + tuple(self._tasks[1])
            polling = any(self._needs_poll(job) for job in pending)
            with self._condition:
                if pending and generation == self._wake_generation:
                    self._condition.wait(timeout=0.001 if polling else None)

    def _run_cleanup(self):
        self._bind_device()
        while True:
            with self._condition:
                while not self._cleanup:
                    if self._closing and not self._thread.is_alive():
                        return
                    self._condition.wait(timeout=0.001 if self._closing else None)
                job = self._cleanup.popleft()
            try:
                job.task()
            except Exception as exc:
                job.error = exc
            finally:
                job.done.set()
                self.wake()

    def close(self, timeout=120.0):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        deadline = time.monotonic() + timeout
        for thread in (self._thread, self._cleanup_thread):
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                raise TimeoutError("read service still owns device work; backing retained")
