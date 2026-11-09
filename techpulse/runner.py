"""The polling loop: nothing a poll raises may end the job."""
import logging
import time

log = logging.getLogger("techpulse")

MAX_FAILURE_WAIT_SECONDS = 1800


def run_forever(name, poll, interval, stop, sleep=None, clock=time.monotonic):
    """Call poll every `interval` seconds, waiting exponentially longer after failures."""
    sleep = sleep or stop.wait
    failures = 0
    while not stop.is_set():
        started = clock()
        try:
            stats = poll()
        except Exception:
            failures += 1
            wait = min(interval * 2 ** (failures - 1), MAX_FAILURE_WAIT_SECONDS)
            log.exception("%s poll failed (%d in a row), next try in %ss", name, failures, wait)
        else:
            failures = 0
            wait = interval
            log.info("%s poll ok: %s", name, stats)
        if not stop.is_set():
            sleep(max(0, wait - (clock() - started)))
