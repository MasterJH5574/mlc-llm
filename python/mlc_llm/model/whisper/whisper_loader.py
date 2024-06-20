"""
This file specifies how MLC's Whisper parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .whisper_model import WhisperConfig, WhisperForConditionalGeneration


def huggingface(model_config: WhisperConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : WhisperConfig
        The configuration of the GPTNeoX model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = WhisperForConditionalGeneration(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for mlc_name, mlc_param in named_parameters.items():
        mapping.add_mapping(
            mlc_name,
            [mlc_name],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )
    return mapping
