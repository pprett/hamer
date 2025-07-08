from typing import Any, Dict, List, Optional, Union
from multiprocessing import resource_tracker

import numpy as np

from robostate import context as _context
from robostate.shared_queue import SharedMemoryQueue, SharedAtomicCounter, SharedNDArray


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked
    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)

    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


# Installing monkey-patch to remove shared memory from resource tracker
remove_shm_from_resource_tracker()


class _SerializableSharedMemoryQueue(SharedMemoryQueue):
    """A serializable version of SharedMemoryQueue.


    This class is used to serialize SharedMemoryQueue objects for storage in the shared memory.

    TODO eventually this should move upstream to robostate.
    """

    def __init__(
        self,
        storage_server,
        array_specs: List[_context.ArraySpec],
        buffer_size: int,
    ):
        shm_manager = storage_server.shmem_m
        super().__init__(shm_manager, array_specs, buffer_size)

    def __getstate__(self) -> dict:
        return {
            "buffer_size": self.buffer_size,
            "array_specs": [
                {
                    "name": spec.name,
                    "dtype": spec.dtype.str,
                    "shape": list(spec.shape),
                }
                for spec in self.array_specs
            ],
            "write_counter": self.write_counter.__getstate__(),
            "read_counter": self.read_counter.__getstate__(),
            "shared_arrays": {
                key: array.__getstate__() for key, array in self.shared_arrays.items()
            },
        }

    def __setstate__(self, state: dict):
        self.buffer_size = state["buffer_size"]
        self.array_specs = [
            _context.ArraySpec(
                spec["name"],
                tuple(spec["shape"]),
                np.dtype(spec["dtype"]),
            )
            for spec in state["array_specs"]
        ]
        self.read_counter = SharedAtomicCounter.from_state(state["read_counter"])
        self.write_counter = SharedAtomicCounter.from_state(state["write_counter"])
        self.shared_arrays = {}
        shared_arrays = state["shared_arrays"]
        for spec in self.array_specs:
            self.shared_arrays[spec.name] = SharedNDArray.from_state(
                shared_arrays[spec.name]
            )

    @classmethod
    def from_state(cls, state) -> "_SerializableSharedMemoryQueue":
        obj = cls.__new__(cls)
        obj.__setstate__(state)
        return obj


class Context(_context.StorageContext):
    """
    Context manager for the robotstate.StorageContext.

    TODO: we should move this upstream to robostate eventually.
    """

    def __enter__(self):
        if self._is_online:
            return self
        else:
            start_server = False
            try:
                self.connect(print_exc=False)
                start_server = not self._is_online
            except (ConnectionError, ConnectionRefusedError):
                start_server = True
            if start_server:
                self._is_online = self.start_server(block=False)
                # This is required with the lazy load change to handle different mp start contexts
                self.m.StorageServer().start()
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._is_online:
            # FIXME robostate Context should have a proper shutdown; only started MpProcess objects have a shutdown finalizer.
            if hasattr(self.m, "shutdown"):
                self.m.shutdown()

        self._is_online = False

    def StaticQueueState(
        self,
        topic_name: str,
        buffer_size: int,
        array_specs: Optional[list[_context.ArraySpec]] = None,
        examples: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> _SerializableSharedMemoryQueue:
        assert self._is_online
        assert (
            array_specs is not None or examples is not None
        ), "Either array_specs or examples must be provided."
        if examples is not None:
            array_specs = _context.array_specs_from_examples(examples)
        state = self.m.StorageServer().make_topic(
            topic_name,
            _SerializableSharedMemoryQueue,
            array_specs=array_specs,
            buffer_size=buffer_size,
            **kwargs,
        )
        return _SerializableSharedMemoryQueue.from_state(state)

    def get_state(
        self, topic_name: str
    ) -> (
        _context.SharedMemoryDynamicRingBuffer
        | _context.SharedMemoryRingBuffer
        | _SerializableSharedMemoryQueue
        | None
    ):
        assert self._is_online
        state = self.m.StorageServer().get_topic(topic_name, lock=True)

        if state is None:
            return None

        if state["type"] == _context.SharedMemoryRingBuffer.__name__:
            return _context.SharedMemoryRingBuffer.from_state(state)
        elif state["type"] == _context.SharedMemoryDynamicRingBuffer.__name__:
            return _context.SharedMemoryDynamicRingBuffer.from_state(
                self.m.StorageServer(), state
            )
        elif state["type"] == _SerializableSharedMemoryQueue.__name__:
            return _SerializableSharedMemoryQueue.from_state(state)
        else:
            raise NotImplementedError(f"Unknown state type {state['type']}.")

    def get_states(
        self,
    ) -> Dict[
        str,
        Union[
            _context.SharedMemoryDynamicRingBuffer,
            _context.SharedMemoryRingBuffer,
            _SerializableSharedMemoryQueue,
            None,
        ],
    ]:
        assert self._is_online
        return {
            topic_name: self.get_state(topic_name)
            for topic_name in self.m.StorageServer().get_topics()
        }


if __name__ == "__main__":
    with Context() as ctx:
        for topic, topic_info in ctx.get_states().items():
            print(topic)
