"""
Test progress tracking functionality with multiprocessing.

These tests verify that the progress counter infrastructure works correctly
across process boundaries on all platforms (both fork and spawn modes).
"""

import multiprocessing
import pickle
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from nmaipy.api_common import PROGRESS_BATCH_SIZE, BaseApiClient


class _ProgressOnlyClient:
    """Minimal stub exposing BaseApiClient's progress methods.

    Constructing a full FeatureApi/BaseApiClient requires API key plumbing
    and pulls in unrelated init. This stub binds only the attributes the
    progress methods actually read, so we can exercise the real batched
    code path without standing up the rest of the client.
    """

    _increment_progress = BaseApiClient._increment_progress
    flush_progress = BaseApiClient.flush_progress

    def __init__(self, progress_counters):
        self._thread_local = threading.local()
        self.progress_counters = progress_counters
        self._progress_buffers = []
        self._progress_buffers_registry_lock = threading.Lock()


def worker_increment_counter(progress_counters, num_increments=5, simulate_gridding=False):
    """
    Worker function that simulates API requests by incrementing counters.

    Args:
        progress_counters: Shared dict with 'total', 'completed', and 'lock'
        num_increments: Number of times to increment completed counter
        simulate_gridding: If True, add extra requests to total (simulating gridding)
    """
    # Simulate gridding by adding more to total
    if simulate_gridding:
        with progress_counters["lock"]:
            progress_counters["total"] += 10
        time.sleep(0.1)  # Simulate gridding overhead

    # Simulate processing requests
    for _ in range(num_increments):
        time.sleep(0.05)  # Simulate network latency
        with progress_counters["lock"]:
            progress_counters["completed"] += 1

    return True


def test_progress_counters_basic():
    """Test that progress counters can be created and accessed"""
    manager = multiprocessing.Manager()
    progress_counters = manager.dict({"total": 10, "completed": 0, "lock": manager.Lock()})

    # Test basic access
    assert progress_counters["total"] == 10
    assert progress_counters["completed"] == 0

    # Test locked increment
    with progress_counters["lock"]:
        progress_counters["completed"] += 1

    assert progress_counters["completed"] == 1


def test_progress_counters_with_process_pool():
    """Test that progress counters work correctly with ProcessPoolExecutor"""
    # Create shared progress counters
    manager = multiprocessing.Manager()
    progress_counters = manager.dict(
        {
            "total": 20,  # 4 workers * 5 increments each
            "completed": 0,
            "lock": manager.Lock(),
        }
    )

    # Submit work to process pool
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for i in range(4):
            future = executor.submit(
                worker_increment_counter,
                progress_counters,
                num_increments=5,
                simulate_gridding=False,
            )
            futures.append(future)

        # Wait for all to complete
        for future in futures:
            assert future.result() == True

    # Verify counters
    assert progress_counters["completed"] == 20
    assert progress_counters["total"] == 20


def test_progress_counters_with_gridding():
    """Test that progress counters handle dynamic total growth (gridding simulation)"""
    # Create shared progress counters
    manager = multiprocessing.Manager()
    progress_counters = manager.dict(
        {
            "total": 10,  # Initial estimate: 2 workers * 5 increments
            "completed": 0,
            "lock": manager.Lock(),
        }
    )

    # Submit work to process pool, one worker will simulate gridding
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []

        # First worker: normal processing
        futures.append(
            executor.submit(
                worker_increment_counter,
                progress_counters,
                num_increments=5,
                simulate_gridding=False,
            )
        )

        # Second worker: simulates gridding (adds 10 to total)
        futures.append(
            executor.submit(
                worker_increment_counter,
                progress_counters,
                num_increments=5,
                simulate_gridding=True,
            )
        )

        # Wait for all to complete
        for future in futures:
            assert future.result() == True

    # Verify counters: total should have grown by 10 due to "gridding"
    assert progress_counters["completed"] == 10
    assert progress_counters["total"] == 20  # 10 initial + 10 from gridding


def test_progress_counters_concurrent_access():
    """Test that concurrent access to counters is thread-safe"""
    manager = multiprocessing.Manager()
    progress_counters = manager.dict({"total": 100, "completed": 0, "lock": manager.Lock()})

    # Submit many small jobs concurrently
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(20):  # 20 workers * 5 increments = 100 total
            future = executor.submit(
                worker_increment_counter,
                progress_counters,
                num_increments=5,
                simulate_gridding=False,
            )
            futures.append(future)

        # Wait for all to complete
        for future in futures:
            assert future.result() == True

    # Verify no increments were lost due to race conditions
    assert progress_counters["completed"] == 100


def test_progress_counters_picklable():
    """Test that progress counters can be pickled (required for spawn mode)"""
    manager = multiprocessing.Manager()
    progress_counters = manager.dict({"total": 10, "completed": 0, "lock": manager.Lock()})

    # Verify it can be pickled (required for passing to worker processes on macOS)
    try:
        pickled = pickle.dumps(progress_counters)
        unpickled = pickle.loads(pickled)
        assert unpickled["total"] == 10
        assert unpickled["completed"] == 0
    except Exception as e:
        pytest.fail(f"Progress counters should be picklable: {e}")


def _writer_blast(progress_counters, num_threads, increments_per_thread):
    """Worker that runs num_threads × increments_per_thread plain `lock + +=1`
    increments — the *unbatched* worst case for lock contention. Used by the
    blocking-reader regression test below.
    """

    def thread_loop():
        for _ in range(increments_per_thread):
            with progress_counters["lock"]:
                progress_counters["completed"] += 1
            time.sleep(0.0005)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(thread_loop) for _ in range(num_threads)]
        for f in futures:
            f.result()
    return True


def test_blocking_reader_not_starved_under_burst_writes():
    """Regression test for the displayed-counter freeze.

    The previous implementation read the counter via
    ``progress_counters["lock"].acquire(timeout=0.1)`` and skipped the update
    when the bounded timeout expired. Under realistic export load (many
    writer threads across multiple worker processes hammering a shared
    Manager.Lock()) the bounded acquire consistently lost — the displayed
    counter froze at the same value for many minutes despite work continuing,
    catching up only when worker pressure dropped at the end of the run.

    The fix switches the monitor to a blocking acquire (no timeout). It may
    wait a few seconds during burst windows but always eventually gets a
    fresh value. This test drives the worst-case unbatched write pattern
    from multiple processes and asserts the reader (using blocking acquire)
    observes many intermediate values and the final tally is exact.
    """
    manager = multiprocessing.Manager()
    NUM_WRITER_PROCESSES = 4
    THREADS_PER_PROCESS = 20
    INC_PER_THREAD = 50
    expected = NUM_WRITER_PROCESSES * THREADS_PER_PROCESS * INC_PER_THREAD
    progress_counters = manager.dict({"total": expected, "completed": 0, "lock": manager.Lock()})

    observed: list = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            with progress_counters["lock"]:
                observed.append(progress_counters["completed"])
            time.sleep(0.02)
        with progress_counters["lock"]:
            observed.append(progress_counters["completed"])

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    try:
        with ProcessPoolExecutor(max_workers=NUM_WRITER_PROCESSES) as pool:
            futures = [
                pool.submit(_writer_blast, progress_counters, THREADS_PER_PROCESS, INC_PER_THREAD)
                for _ in range(NUM_WRITER_PROCESSES)
            ]
            for f in futures:
                assert f.result() is True
    finally:
        stop.set()
        reader_thread.join()

    assert progress_counters["completed"] == expected
    distinct = sorted(set(observed))
    assert observed == sorted(observed), "counter must be monotonic from reader's view"
    assert len(distinct) >= 20, (
        f"reader observed only {len(distinct)} distinct values "
        f"({distinct[:5]}..{distinct[-5:]}) — blocking-acquire reader appears starved"
    )


def test_flush_progress_drains_residue_per_thread_buffers():
    """flush_progress must drain pending per-thread buffers into the shared counter.

    Forces every opportunistic flush in `_increment_progress` to fail by
    holding the shared lock externally for the duration of the writes. With
    per-thread increments kept below PROGRESS_BATCH_SIZE, the forced flush
    inside `_increment_progress` never fires either, so all 50 increments
    end up trapped in per-thread buffers. The shared counter stays at 0
    until `flush_progress()` runs.

    This is the exact failure mode that produces "tqdm bar jumps to 100%
    at the end" on real exports — and the test that protects the fix
    from regressing.
    """
    manager = multiprocessing.Manager()
    progress_counters = manager.dict({"total": 100, "completed": 0, "lock": manager.Lock()})
    client = _ProgressOnlyClient(progress_counters)

    shared_lock = progress_counters["lock"]
    held = threading.Event()
    release = threading.Event()

    def holder():
        shared_lock.acquire()
        held.set()
        release.wait()
        shared_lock.release()

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    held.wait()

    NUM_THREADS = 5
    # Stay strictly below PROGRESS_BATCH_SIZE so the forced flush inside
    # _increment_progress never fires (it would block on the held lock).
    INC_PER_THREAD = PROGRESS_BATCH_SIZE - 1
    expected = NUM_THREADS * INC_PER_THREAD

    try:

        def worker():
            for _ in range(INC_PER_THREAD):
                client._increment_progress()

        threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        release.set()
        holder_thread.join()

    # All increments are trapped in per-thread buffers — shared counter is still 0.
    assert progress_counters["completed"] == 0, (
        f"expected all {expected} increments to be buffered, but shared counter is " f"{progress_counters['completed']}"
    )

    client.flush_progress()

    # After flush: every increment is accounted for.
    assert progress_counters["completed"] == expected

    # And buffers are empty (calling flush again is a no-op).
    client.flush_progress()
    assert progress_counters["completed"] == expected


def test_increment_progress_eventually_consistent_under_contention():
    """Under realistic contention, batched writers + per-call opportunistic flush
    eventually account for every increment after flush_progress.

    Doesn't assert anything about the running drift (timing-dependent) — only
    that the post-flush total is exact. Validates the batched-writer path
    end-to-end through the actual BaseApiClient methods.
    """
    manager = multiprocessing.Manager()
    NUM_THREADS = 12
    INC_PER_THREAD = 200
    expected = NUM_THREADS * INC_PER_THREAD
    progress_counters = manager.dict({"total": expected, "completed": 0, "lock": manager.Lock()})
    client = _ProgressOnlyClient(progress_counters)

    def worker():
        for _ in range(INC_PER_THREAD):
            client._increment_progress()

    threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    client.flush_progress()
    assert progress_counters["completed"] == expected


def test_progress_counters_with_skipped_work():
    """Test that progress counters handle pre-existing work correctly (resume scenario)"""
    # Simulate a scenario where we have 20 total AOIs, but 10 were already processed
    # This mimics resuming after a crash where some chunks already exist
    already_completed = 10
    remaining_work = 10

    manager = multiprocessing.Manager()
    progress_counters = manager.dict(
        {
            "total": remaining_work,  # Only count work that needs to be done
            "completed": 0,
            "lock": manager.Lock(),
        }
    )

    # Submit only the remaining work
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for i in range(2):  # 2 workers * 5 = 10 remaining
            future = executor.submit(
                worker_increment_counter,
                progress_counters,
                num_increments=5,
                simulate_gridding=False,
            )
            futures.append(future)

        for future in futures:
            assert future.result() == True

    # Verify: should show 100% completion for the remaining work
    assert progress_counters["completed"] == 10
    assert progress_counters["total"] == 10
    # Total work was 20, but we only tracked the 10 that needed to be done
