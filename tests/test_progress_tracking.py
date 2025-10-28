"""
Test progress tracking functionality with multiprocessing.

These tests verify that the progress counter infrastructure works correctly
across process boundaries on all platforms (both fork and spawn modes).
"""
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
import pytest


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
        with progress_counters['lock']:
            progress_counters['total'] += 10
        time.sleep(0.1)  # Simulate gridding overhead

    # Simulate processing requests
    for _ in range(num_increments):
        time.sleep(0.05)  # Simulate network latency
        with progress_counters['lock']:
            progress_counters['completed'] += 1

    return True


def test_progress_counters_basic():
    """Test that progress counters can be created and accessed"""
    manager = multiprocessing.Manager()
    progress_counters = manager.dict({
        'total': 10,
        'completed': 0,
        'lock': manager.Lock()
    })

    # Test basic access
    assert progress_counters['total'] == 10
    assert progress_counters['completed'] == 0

    # Test locked increment
    with progress_counters['lock']:
        progress_counters['completed'] += 1

    assert progress_counters['completed'] == 1


def test_progress_counters_with_process_pool():
    """Test that progress counters work correctly with ProcessPoolExecutor"""
    # Create shared progress counters
    manager = multiprocessing.Manager()
    progress_counters = manager.dict({
        'total': 20,  # 4 workers * 5 increments each
        'completed': 0,
        'lock': manager.Lock()
    })

    # Submit work to process pool
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for i in range(4):
            future = executor.submit(
                worker_increment_counter,
                progress_counters,
                num_increments=5,
                simulate_gridding=False
            )
            futures.append(future)

        # Wait for all to complete
        for future in futures:
            assert future.result() == True

    # Verify counters
    assert progress_counters['completed'] == 20
    assert progress_counters['total'] == 20


def test_progress_counters_with_gridding():
    """Test that progress counters handle dynamic total growth (gridding simulation)"""
    # Create shared progress counters
    manager = multiprocessing.Manager()
    progress_counters = manager.dict({
        'total': 10,  # Initial estimate: 2 workers * 5 increments
        'completed': 0,
        'lock': manager.Lock()
    })

    # Submit work to process pool, one worker will simulate gridding
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []

        # First worker: normal processing
        futures.append(executor.submit(
            worker_increment_counter,
            progress_counters,
            num_increments=5,
            simulate_gridding=False
        ))

        # Second worker: simulates gridding (adds 10 to total)
        futures.append(executor.submit(
            worker_increment_counter,
            progress_counters,
            num_increments=5,
            simulate_gridding=True
        ))

        # Wait for all to complete
        for future in futures:
            assert future.result() == True

    # Verify counters: total should have grown by 10 due to "gridding"
    assert progress_counters['completed'] == 10
    assert progress_counters['total'] == 20  # 10 initial + 10 from gridding


def test_progress_counters_concurrent_access():
    """Test that concurrent access to counters is thread-safe"""
    manager = multiprocessing.Manager()
    progress_counters = manager.dict({
        'total': 100,
        'completed': 0,
        'lock': manager.Lock()
    })

    # Submit many small jobs concurrently
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(20):  # 20 workers * 5 increments = 100 total
            future = executor.submit(
                worker_increment_counter,
                progress_counters,
                num_increments=5,
                simulate_gridding=False
            )
            futures.append(future)

        # Wait for all to complete
        for future in futures:
            assert future.result() == True

    # Verify no increments were lost due to race conditions
    assert progress_counters['completed'] == 100


def test_progress_counters_picklable():
    """Test that progress counters can be pickled (required for spawn mode)"""
    import pickle

    manager = multiprocessing.Manager()
    progress_counters = manager.dict({
        'total': 10,
        'completed': 0,
        'lock': manager.Lock()
    })

    # Verify it can be pickled (required for passing to worker processes on macOS)
    try:
        pickled = pickle.dumps(progress_counters)
        unpickled = pickle.loads(pickled)
        assert unpickled['total'] == 10
        assert unpickled['completed'] == 0
    except Exception as e:
        pytest.fail(f"Progress counters should be picklable: {e}")


def test_progress_counters_with_skipped_work():
    """Test that progress counters handle pre-existing work correctly (resume scenario)"""
    # Simulate a scenario where we have 20 total AOIs, but 10 were already processed
    # This mimics resuming after a crash where some chunks already exist
    already_completed = 10
    remaining_work = 10

    manager = multiprocessing.Manager()
    progress_counters = manager.dict({
        'total': remaining_work,  # Only count work that needs to be done
        'completed': 0,
        'lock': manager.Lock()
    })

    # Submit only the remaining work
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for i in range(2):  # 2 workers * 5 = 10 remaining
            future = executor.submit(
                worker_increment_counter,
                progress_counters,
                num_increments=5,
                simulate_gridding=False
            )
            futures.append(future)

        for future in futures:
            assert future.result() == True

    # Verify: should show 100% completion for the remaining work
    assert progress_counters['completed'] == 10
    assert progress_counters['total'] == 10
    # Total work was 20, but we only tracked the 10 that needed to be done


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running progress tracking tests...")
    test_progress_counters_basic()
    print("✓ Basic test passed")

    test_progress_counters_with_process_pool()
    print("✓ Process pool test passed")

    test_progress_counters_with_gridding()
    print("✓ Gridding simulation test passed")

    test_progress_counters_concurrent_access()
    print("✓ Concurrent access test passed")

    test_progress_counters_picklable()
    print("✓ Picklable test passed")

    test_progress_counters_with_skipped_work()
    print("✓ Skipped work test passed")

    print("\nAll tests passed!")
