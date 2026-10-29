from types import SimpleNamespace
import threading

from tutti.engine.read_service import ReadService


def _service(monkeypatch):
    monkeypatch.setattr(ReadService, "_bind_device", lambda self: None)
    return ReadService(SimpleNamespace())


def test_foreground_runs_before_queued_background(monkeypatch):
    service = _service(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    order = []

    class Blocker:
        def advance(self):
            entered.set()
            assert release.wait(2)
            return True

    class Task:
        def __init__(self, name):
            self.name = name

        def advance(self):
            order.append(self.name)
            return True

    blocker = service.submit(Blocker())
    assert entered.wait(2)
    background = service.submit(Task("background"), priority=1)
    foreground = service.submit(Task("foreground"))
    release.set()
    for job in (blocker, background, foreground):
        job.join(2)
        assert not job.is_alive()
    service.close()
    assert order == ["foreground", "background"]


def test_tasks_reuse_workers_and_cleanup_does_not_block_submission(monkeypatch):
    service = _service(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    identities = []

    def cleanup():
        entered.set()
        assert release.wait(2)

    class Task:
        def advance(self):
            identities.append(threading.get_ident())
            return True

    pending = service.submit_cleanup(cleanup)
    assert entered.wait(2)
    for _ in range(3):
        job = service.submit(Task())
        job.join(2)
        assert not job.is_alive()
    assert len(set(identities)) == 1
    assert pending.is_alive()
    release.set()
    service.close()
    assert not pending.is_alive()


def test_background_lease_keeps_fifo_across_pauses_and_cancellation(monkeypatch):
    service = _service(monkeypatch)
    first, second, third = object(), object(), object()
    assert service.acquire_background(first)
    assert not service.acquire_background(second)
    assert not service.acquire_background(third)
    assert service.acquire_background(first)
    service.release_background(second)
    service.release_background(first)
    assert service.acquire_background(third)
    service.release_background(third)
    service.close()


def test_paused_task_waits_for_notification(monkeypatch):
    service = _service(monkeypatch)
    advanced = threading.Event()
    allowed = threading.Event()
    calls = []

    class Task:
        def advance(self):
            calls.append(threading.get_ident())
            advanced.set()
            return allowed.is_set()

        def needs_poll(self):
            return False

    job = service.submit(Task(), priority=1)
    assert advanced.wait(2)
    job.join(0.02)
    assert job.is_alive()
    assert len(calls) == 1
    allowed.set()
    service.wake()
    job.join(2)
    assert not job.is_alive()
    service.close()
    assert len(set(calls)) == 1


def test_foreground_fences_join_on_one_reused_independent_stream(monkeypatch):
    import torch
    calls = []

    class Event:
        def record(self, stream):
            calls.append(("record", self, stream))

    class Stream:
        def wait_event(self, event):
            calls.append(("wait", event, self))

    streams = []

    def stream(device):
        result = Stream()
        streams.append((device, result))
        return result

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "Stream", stream)
    service = _service(monkeypatch)
    service._store._accel_id = 5
    fences = (Event(), Event())
    first = service.join_events(fences)
    second = service.join_events((Event(),))
    service.close()
    assert len(streams) == 1 and streams[0][0] == 5
    assert first is not second
    assert [item[0] for item in calls] == ["wait", "wait", "record", "wait", "record"]
    assert all(item[2] is streams[0][1] for item in calls)
    assert service.join_events((object(),)) is None


def test_task_failure_does_not_stop_service(monkeypatch):
    service = _service(monkeypatch)
    errors = []

    class Failed:
        def advance(self):
            raise RuntimeError("failed read")

        def fail(self, error):
            errors.append(str(error))

    class Success:
        def advance(self):
            return True

    failed = service.submit(Failed())
    failed.join(2)
    success = service.submit(Success())
    success.join(2)
    service.close()
    assert errors == ["failed read"]
    assert isinstance(failed.error, RuntimeError)
    assert not success.is_alive()
