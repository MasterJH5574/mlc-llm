"""MLC Chat python package.

MLC Chat is the app runtime of MLC LLM.
"""

from typing import Tuple

import torch
from tvm import register_global_func

from . import protocol, serve
from .libinfo import __version__
from .serve import AsyncMLCEngine, MLCEngine


@register_global_func("runtime.disco.create_socket_session_local_workers", override=True)
def _create_socket_session_local_workers(num_workers):
    """Create the local session for each distributed node over socket session."""
    from tvm.runtime.disco import (  # pylint: disable=import-outside-toplevel
        ProcessSession,
    )

    return ProcessSession(num_workers, num_groups=1, entrypoint="mlc_llm.cli.worker")


generator = None


@register_global_func("flashinfer.random.get_seed_and_offset", override=True)
def _get_seed_and_offset(increment: int) -> Tuple[int, int]:
    global generator
    if generator is None:
        generator = torch.Generator(device=torch.device("cuda"))
    state = generator.get_state()
    seed, offset = state.view(torch.int64)
    offset += (increment + 3) // 4 * 4
    generator.set_state(torch.tensor([seed, offset], dtype=torch.int64).view(torch.uint8))
    return int(seed), int(offset)
