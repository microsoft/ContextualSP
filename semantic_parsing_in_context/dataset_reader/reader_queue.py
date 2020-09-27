# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import logging
import os
from queue import Empty
from typing import List, Iterable, Iterator, Optional

import numpy as np
from allennlp.data.instance import Instance
from torch.multiprocessing import Process, Queue, Value, log_to_stderr


class logger:
    """
    multiprocessing.log_to_stderr causes some output in the logs
    even when we don't use this dataset reader. This is a small hack
    to instantiate the stderr logger lazily only when it's needed
    (which is only when using the MultiprocessDatasetReader)
    """

    _logger = None

    @classmethod
    def info(cls, message: str) -> None:
        if cls._logger is None:
            cls._logger = log_to_stderr()
            cls._logger.setLevel(logging.INFO)

        cls._logger.info(message)


def _worker(
        call_back,
        input_queue: Queue,
        output_queue: Queue,
        num_active_workers: Value,
        num_inflight_items: Value,
        worker_id: int,
) -> None:
    """
    A worker that pulls filenames off the input queue, uses the dataset reader
    to read them, and places the generated instances on the output queue.  When
    there are no filenames left on the input queue, it decrements
    num_active_workers to signal completion.
    """
    logger.info(f"Reader worker: {worker_id} PID: {os.getpid()}")
    # Keep going until you get a file_path that's None.
    while True:
        file_path = input_queue.get()
        if file_path is None:
            # It's important that we close and join the queue here before
            # decrementing num_active_workers. Otherwise our parent may join us
            # before the queue's feeder thread has passed all buffered items to
            # the underlying pipe resulting in a deadlock.
            #
            # See:
            # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#pipes-and-queues
            # https://docs.python.org/3.6/library/multiprocessing.html?highlight=process#programming-guidelines
            output_queue.close()
            output_queue.join_thread()
            # Decrementing is not atomic.
            # See https://docs.python.org/2/library/multiprocessing.html#multiprocessing.Value.
            with num_active_workers.get_lock():
                num_active_workers.value -= 1
            logger.info(f"Reader worker {worker_id} finished")
            break

        logger.info(f"reading instances from {file_path}")
        instance = call_back(file_path)
        with num_inflight_items.get_lock():
            num_inflight_items.value += 1
        output_queue.put(instance)


class QIterable(Iterable[Instance]):
    """
    You can't set attributes on Iterators, so this is just a dumb wrapper
    that exposes the output_queue.
    """

    def __init__(self, output_queue_size, epochs_per_read, num_workers, call_back, file_path) -> None:
        self.output_queue = Queue(output_queue_size)
        self.epochs_per_read = epochs_per_read
        self.num_workers = num_workers
        self.file_path = file_path

        self.call_back = call_back
        # Initialized in start.
        self.input_queue: Optional[Queue] = None
        self.processes: List[Process] = []
        # The num_active_workers and num_inflight_items counts in conjunction
        # determine whether there could be any outstanding instances.
        self.num_active_workers: Optional[Value] = None
        self.num_inflight_items: Optional[Value] = None

    def __iter__(self) -> Iterator[Instance]:
        self.start()

        # Keep going as long as not all the workers have finished or there are items in flight.
        while self.num_active_workers.value > 0 or self.num_inflight_items.value > 0:
            # Inner loop to minimize locking on self.num_active_workers.
            while True:
                try:
                    # Non-blocking to handle the empty-queue case.
                    yield self.output_queue.get(block=False, timeout=1.0)
                    with self.num_inflight_items.get_lock():
                        self.num_inflight_items.value -= 1
                except Empty:
                    # The queue could be empty because the workers are
                    # all finished or because they're busy processing.
                    # The outer loop distinguishes between these two
                    # cases.
                    break

        self.join()

    def start(self) -> None:
        shards = glob.glob(self.file_path)
        # Ensure a consistent order before shuffling for testing.
        shards.sort()
        num_shards = len(shards)

        # If we want multiple epochs per read, put shards in the queue multiple times.
        self.input_queue = Queue(num_shards * self.epochs_per_read + self.num_workers)
        for _ in range(self.epochs_per_read):
            np.random.shuffle(shards)
            for shard in shards:
                self.input_queue.put(shard)

        # Then put a None per worker to signify no more files.
        for _ in range(self.num_workers):
            self.input_queue.put(None)

        assert (
            not self.processes
        ), "Process list non-empty! You must call QIterable.join() before restarting."
        self.num_active_workers = Value("i", self.num_workers)
        self.num_inflight_items = Value("i", 0)
        for worker_id in range(self.num_workers):
            process = Process(
                target=_worker,
                args=(
                    self.call_back,
                    self.input_queue,
                    self.output_queue,
                    self.num_active_workers,
                    self.num_inflight_items,
                    worker_id,
                ),
            )
            logger.info(f"starting worker {worker_id}")
            process.start()
            self.processes.append(process)

    def join(self) -> None:
        for process in self.processes:
            process.join()
        self.processes.clear()

    def __del__(self) -> None:
        """
        Terminate processes if the user hasn't joined. This is necessary as
        leaving stray processes running can corrupt shared state. In brief,
        we've observed shared memory counters being reused (when the memory was
        free from the perspective of the parent process) while the stray
        workers still held a reference to them.

        For a discussion of using destructors in Python in this manner, see
        https://eli.thegreenplace.net/2009/06/12/safely-using-destructors-in-python/.
        """
        for process in self.processes:
            process.terminate()
