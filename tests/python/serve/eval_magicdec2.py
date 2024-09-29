# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import asyncio
import os
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.serve import AsyncMLCEngine, EngineConfig
from mlc_llm.support import argparse, logging

logging.enable_logging
logger = logging.getLogger(__name__)


def convert_pg19_dataset(dataset_path, tokenizer, seq_len, end=1):
    import os

    import torch

    num_prompts = 60

    d_files = os.listdir(dataset_path)
    dataset = load_dataset(
        "json", data_files=[dataset_path + name for name in d_files], split="train"
    )
    tokenized_prompts = []
    for i in range(0, num_prompts):
        prompt = dataset[i]["text"]
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
        tokenized_prompt = tokenized_prompt[:, 8000:]
        tokenized_prompt = tokenized_prompt.split(seq_len, dim=-1)[:-1]

        for i in range(len(tokenized_prompt)):
            tokenized_prompt[i][:, 0] = tokenizer.bos_token_id
            tokenized_prompts.append(tokenized_prompt[i])
    data = torch.cat(tokenized_prompts, dim=0).repeat(end, 1)
    return data


async def run_benchmark(
    async_engine: AsyncMLCEngine, prompts: List[List[int]], generation_cfg: GenerationConfig
) -> Tuple[List[List[str]], List[List[int]]]:
    num_requests = len(prompts)
    output_deltas: List[List[str]] = [[] for _ in range(num_requests)]
    output_timestamps: List[List[float]] = [[] for _ in range(num_requests)]
    terminated = False

    async def generate_task(
        async_engine: AsyncMLCEngine,
        prompt: List[int],
        generation_cfg: GenerationConfig,
        request_id: str,
    ):
        nonlocal terminated
        rid = int(request_id)
        async for delta_outputs in async_engine._generate(
            prompt, generation_cfg, request_id=request_id
        ):
            assert len(delta_outputs) == 1
            if not terminated and delta_outputs[0].delta_text != "":
                output_deltas[rid].append(delta_outputs[0].delta_text)
                output_timestamps[rid].append(time.time())
        terminated = True

    tasks = [
        asyncio.create_task(
            generate_task(async_engine, prompts[i], generation_cfg, request_id=str(i))
        )
        for i in range(num_requests)
    ]

    await asyncio.gather(*tasks)
    return output_deltas, output_timestamps


def calculate_metrics(
    output_deltas: List[List[str]],
    output_timestamps: List[List[float]],
    tokenizer: AutoTokenizer,
):
    assert len(output_deltas) == len(output_timestamps)
    total_decode_tokens = 0
    for deltas, timestamps in zip(output_deltas, output_timestamps):
        assert len(deltas) == len(timestamps) > 2
        output_text = "".join(deltas)
        discarded_text = deltas[0] + deltas[1]
        num_tokens = len(tokenizer.encode(output_text)) - len(tokenizer.encode(discarded_text))
        total_decode_tokens += num_tokens
    start_time = min(timestamps[1] for timestamps in output_timestamps)
    end_time = max(timestamps[-1] for timestamps in output_timestamps)
    elapsed_time = end_time - start_time
    return total_decode_tokens, elapsed_time


def commit_metrics(file_path: str, metrics: Dict[str, Any]):
    # Check if file exists
    if os.path.exists(file_path):
        try:
            # Load existing DataFrame
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            logger.info(
                "Output file %s is corrupted. Creating a new dataframe and overriding the file.",
                file_path,
            )
            df = pd.DataFrame()
    else:
        # Create a new DataFrame if the file does not exist
        df = pd.DataFrame()
    # Convert the dictionary into a DataFrame with a single row
    new_row = pd.DataFrame([metrics])
    # Append the new row to the DataFrame using pd.concat()
    df = pd.concat([df, new_row], ignore_index=True)
    # Save the updated DataFrame to file
    df.to_csv(file_path, index=False)


async def test_engine_generate(args: argparse.argparse.Namespace):
    # Create engine
    spec_decode_mode = "disable" if args.draft_len == 0 else "magic_dec2"
    async_engine = AsyncMLCEngine(
        model=args.model,
        model_lib=args.model_lib,
        mode="server",
        engine_config=EngineConfig(
            max_num_sequence=args.num_requests,
            speculative_mode=spec_decode_mode,
            spec_draft_length=args.draft_len,
            prefix_cache_mode="disable",
            gpu_memory_utilization=args.gpu_memory_utilization,
            prefill_mode="chunked",
            prefill_chunk_size=8192,
            tensor_parallel_shards=args.tp,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_data = convert_pg19_dataset(args.dataset, tokenizer, seq_len=args.input_len)

    generation_cfg = GenerationConfig(
        max_tokens=args.output_len, top_p=args.top_p, temperature=args.temperature
    )

    assert args.num_requests <= prompt_data.shape[0]
    prompts = [prompt_data[i].tolist() for i in range(args.num_requests)]

    logger.info("Warmup two rounds with %d requests", args.num_requests)
    await run_benchmark(async_engine, prompts, generation_cfg)
    if args.draft_len > 0:
        await run_benchmark(async_engine, prompts, generation_cfg)

    async_engine.reset()
    logger.info("Run benchmark with %d requsets", args.num_requests)
    output_deltas, output_timestamps = await run_benchmark(async_engine, prompts, generation_cfg)

    logger.info("All finished")
    for req_id, output in enumerate(output_deltas):
        print(f"Output {req_id}:{''.join(output)}\n")
    engine_metrics = (await async_engine.metrics()).metrics
    async_engine.terminate()
    del async_engine

    total_decode_tokens, elapsed_time = calculate_metrics(
        output_deltas, output_timestamps, tokenizer
    )
    metrics = {
        "num_requests": args.num_requests,
        "input_length": args.input_len,
        "output_length": args.output_len,
        "draft_length": args.draft_len,
        "draft_budget": args.draft_budget,
        "tp": args.tp,
        "total_decode_tokens": total_decode_tokens,
        "elapsed_time": elapsed_time,
        "decode_throughput": total_decode_tokens / elapsed_time,
        "engine_metrics": engine_metrics,
    }
    logger.info("Metrics: %s", metrics)
    commit_metrics(args.output, metrics)
    logger.info("Metrics appended to file %s", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLC LLM MagicDec2")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-lib", type=str)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="workspace/pg19/")
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--draft-len", type=int, required=True)
    parser.add_argument("--draft-budget", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output", "-o", type=str, default="mlc_magicdec2.csv")
    args = parser.parse_args()
    logger.info("Benchmark arguments: %s", args)
    asyncio.run(test_engine_generate(args))
