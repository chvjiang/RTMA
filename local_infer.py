import codecs as cs
import json
import os
import os.path as osp
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from hymotion.utils.path import parse_dirs_and_sort
from hymotion.utils.t2m_runtime import T2MRuntime

import torch
import numpy as np
from torch import Tensor

def generate_random_seeds(num_seeds: int = 4) -> List[int]:
    """Generate random seeds."""
    return [random.randint(0, 999) for _ in range(num_seeds)]


def load_ref_motion(path: str, fps: float, runtime: "T2MRuntime") -> Tensor:
    """Load a reference motion for TM2M inference.

    Returns a tensor of shape (1, Lr, 201) on the first pipeline's device,
    normalized with the pipeline's mean/std.
    """
    # ---- 1. read raw motion ----
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        if "motion" in data:
            data = data["motion"]
        else:
            # fall back to the first array if the key name differs
            data = data[list(data.keys())[0]]
    elif path.endswith(".pkl"):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported ref motion format: {path}")

    ref = torch.as_tensor(np.asarray(data), dtype=torch.float32)
    if ref.ndim == 2:
        ref = ref.unsqueeze(0)          # (1, L, D)
    assert ref.ndim == 3, f"ref motion should be 2D or 3D, got {ref.shape}"
    assert ref.shape[-1] == 201, f"ref motion last dim should be 201, got {ref.shape[-1]}"

    # ---- 2. fps alignment (model internal fps is 30) ----
    # model_fps = 30.0
    # if abs(fps - model_fps) > 1e-6:
    #     L = ref.shape[1]
    #     new_L = max(1, int(round(L * model_fps / fps)))
    #     # (1, L, D) -> (1, D, L) -> interpolate -> (1, L, D)
    #     x = ref.permute(0, 2, 1)
    #     x = torch.nn.functional.interpolate(x, size=new_L, mode="linear", align_corners=False)
    #     ref = x.permute(0, 2, 1).contiguous()

    # ---- 3. normalize with the pipeline's mean/std ----
    pipeline = runtime.pipelines[0]
    if hasattr(pipeline, "mean") and hasattr(pipeline, "std"):
        mean = pipeline.mean.to(ref.device).view(1, 1, -1)
        std = pipeline.std.to(ref.device).view(1, 1, -1)
        # avoid division by zero, same convention as decode_motion_from_latent
        std = torch.where(std < 1e-3, torch.ones_like(std), std)
        ref = (ref - mean) / std
    else:
        print(">>> [WARN] pipeline has no mean/std; ref motion is NOT normalized")

    # ---- 4. move to the pipeline's device ----
    device = next(pipeline.parameters()).device
    # return ref.to(device)
    return ref[:150].to(device)


def process_single_text(
    runtime: T2MRuntime,
    text: str,
    duration: float,
    seeds: List[int],
    cfg_scale: float,
    output_dir: str,
    output_filename: str,
    disable_rewrite: bool = False,
    disable_duration_est: bool = False,
    ref_motion: Optional[Tensor] = None,
) -> dict:
    print(f">>> Processing text: {text}")

    call_llm = not disable_rewrite or not disable_duration_est
    if not call_llm:
        print(f"\t>>> Using original duration and original text...")
        predicted_duration = duration
        rewritten_text = text
    else:
        print(f"\t>>> Using LLM to estimate duration/rewrite text...")
        predicted_duration, rewritten_text = runtime.rewrite_text_and_infer_time(text=text)
        if disable_rewrite:
            rewritten_text = text
        if disable_duration_est:
            predicted_duration = duration

    print(f"\t>>> Generating motion: {rewritten_text}")
    seeds_csv = ",".join(map(str, seeds))

    fbx_ok = getattr(runtime, "fbx_available", False)
    req_format = "fbx" if fbx_ok else "dict"
    html, fbx_files, _ = runtime.generate_motion(
        text=rewritten_text,
        seeds_csv=seeds_csv,
        duration=predicted_duration,
        cfg_scale=cfg_scale,
        output_format=req_format,
        original_text=text,
        output_dir=output_dir,
        output_filename=output_filename,
        ref_motion=ref_motion,
    )

    return {
        "text": text,
        "rewritten_text": rewritten_text,
        "duration": predicted_duration,
        "seeds": seeds,
        "file_or_html": fbx_files if fbx_ok else [],
    }


def run_parallel_tasks(
    runtime: T2MRuntime,
    tasks: List[dict],
    cfg_scale: float,
    disable_rewrite: bool = False,
    disable_duration_est: bool = False,
    max_workers: Optional[int] = None,
    ref_motion: Optional[Tensor] = None,
) -> dict:
    """Parallel execution of a standardized task list.

    Task fields requirements:
      - prompt: str
      - duration: float
      - seeds: List[int]
      - output_dir: str
      - output_filename: str (formatted_idx)
    """
    results = {
        "total": len(tasks),
        "success": 0,
        "failed": 0,
        "details": [],
        "saved_files": [],
    }

    def _run_one(task: dict):
        return task["output_filename"], process_single_text(
            runtime=runtime,
            text=task["prompt"],
            duration=task["duration"],
            seeds=task["seeds"],
            cfg_scale=cfg_scale,
            output_dir=task["output_dir"],
            output_filename=task["output_filename"],
            disable_rewrite=disable_rewrite,
            disable_duration_est=disable_duration_est,
            ref_motion=ref_motion,
        )

    if max_workers is None:
        max_workers = max(1, len(runtime.device_ids) if runtime.device_ids else 1)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            try:
                print(f">>> Processing task successfully")
                formatted_idx, result = fut.result()
                results["success"] += 1
                results["details"].append({"formatted_idx": formatted_idx, "status": "success", **result})
                results["saved_files"].extend(result["file_or_html"])
            except Exception as e:
                print(f">>> Processing task failed, {str(e)}")
                i = futures[fut]
                results["failed"] += 1
                results["details"].append({"task_index": i, "error": str(e), "status": "failed"})
    return results


def process_input_file(
    runtime: T2MRuntime,
    input_file: str,
    output_dir: str,
    cfg_scale: Optional[float] = None,
    disable_rewrite: bool = False,
    disable_duration_est: bool = False,
    num_seeds: int = 4,
    ref_motion: Optional[Tensor] = None,
) -> dict:
    """
    Compatible txt / json, parse to unified task list and execute in parallel.
    # The json/txt input file formats can be as follows:
    #
    # For txt files:
    #   Each line represents a task. The format for each line is:
    #     prompt_text[#duration][#unique_id]
    #   Examples:
    #     A man is walking on the beach.#60#001
    #     A woman is running.
    #   If duration is not specified, the default is 150 frames (~5 seconds).
    #   The unique_id is optional and can be used for naming output files.
    #
    # For json files:
    #   The file should be a dictionary where each key is a category or group,
    #   and its value is a list of prompt lines (with the same format as in txt).
    #   Example:
    #   {
    #     "test": [
    #       "A man is dancing #30",
    #       "A person jumps #60"
    #     ],
    #   }
    #   Each entry in the value list follows the same line format as the txt file.
    #   The output can be organized into subdirectories based on the keys.
    # Note: To use fixed duration values, you must specify --disable_duration_est.
    """
    print(f">>> Processing file: {input_file}")
    basename = os.path.basename(input_file).split(".")[0]
    cfg_scale = cfg_scale or 5.0

    results = {
        "input_file": input_file,
        "basename": basename,
        "total": 0,
        "success": 0,
        "failed": 0,
        "details": [],
        "saved_files": [],
    }

    tasks: List[dict] = []
    if input_file.endswith(".txt"):
        with cs.open(input_file, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
        for itext, text_line in enumerate(lines):
            if not text_line:
                continue
            split_list = text_line.split("#")
            prompt = split_list[0].strip()
            length = int(split_list[1]) if len(split_list) > 1 else 100
            test_time = length / 30.0
            orig_fileidx = split_list[2] if len(split_list) > 2 else f"{itext}"
            save_orig_fileidx = int(re.sub(r"\s+", "", orig_fileidx.replace(".", "_").replace("/", "__")))
            formatted_idx = f"{save_orig_fileidx:08d}"
            tasks.append(
                {
                    "prompt": prompt,
                    "duration": test_time,
                    "seeds": generate_random_seeds(num_seeds),
                    "output_dir": output_dir,
                    "output_filename": formatted_idx,
                }
            )
    elif input_file.endswith(".json"):
        with cs.open(input_file, encoding="utf-8") as f:
            text_map = json.load(f)
        for key, value in text_map.items():
            if "_chn" in key or "GENERATE_PROMPT_FORMAT" in key:
                continue
            subdir = osp.join(output_dir, key)
            os.makedirs(subdir, exist_ok=True)
            for itext, text_line in enumerate(value):
                split_list = text_line.strip().split("#")
                prompt = split_list[0].strip()
                length = int(split_list[1]) if len(split_list) > 1 else 100
                test_time = length / 30.0
                orig_fileidx = split_list[2] if len(split_list) > 2 else f"{itext}"
                save_orig_fileidx = int(
                    re.sub(
                        r"\s+",
                        "",
                        orig_fileidx.replace(".", "_").replace("/", "__"),
                    )
                )
                formatted_idx = f"{save_orig_fileidx:08d}"
                tasks.append(
                    {
                        "prompt": prompt,
                        "duration": test_time,
                        "seeds": generate_random_seeds(num_seeds),
                        "output_dir": subdir,
                        "output_filename": formatted_idx,
                    }
                )
    else:
        raise ValueError(f">>> Unsupported file type: {input_file}")

    results["total"] = len(tasks)
    if results["total"] == 0:
        return results

    par_ret = run_parallel_tasks(
        runtime=runtime,
        tasks=tasks,
        cfg_scale=cfg_scale,
        disable_rewrite=disable_rewrite,
        disable_duration_est=disable_duration_est,
        max_workers=max(1, len(runtime.device_ids) if runtime.device_ids else 1),
        ref_motion=ref_motion,
    )
    results.update(
        {
            "success": par_ret["success"],
            "failed": par_ret["failed"],
            "details": par_ret["details"],
            "saved_files": par_ret["saved_files"],
        }
    )
    return results


def save_batch_results(results_list: List[dict], output_dir: str):
    """Save batch processing results."""
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # save detailed results
    results_file = os.path.join(output_dir, f"batch_results_{timestamp}.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)

    # save summary
    total_files = len(results_list)
    total_texts = sum(r["total"] for r in results_list)
    total_success = sum(r["success"] for r in results_list)
    total_failed = sum(r["failed"] for r in results_list)

    summary_file = os.path.join(output_dir, f"batch_summary_{timestamp}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Batch processing summary - {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of processed files: {total_files}\n")
        f.write(f"Total number of texts: {total_texts}\n")
        f.write(f"Number of successful tasks: {total_success}\n")
        f.write(f"Number of failed tasks: {total_failed}\n")
        f.write(f"Success rate: {total_success/total_texts*100:.1f}%\n\n")

        for result in results_list:
            f.write(f"File: {result['basename']}\n")
            f.write(f"  Total: {result['total']}, Success: {result['success']}, Failed: {result['failed']}\n")
            f.write(f"  Number of saved files: {len(result['saved_files'])}\n\n")

    print(f">>> Results saved to: {results_file}")
    print(f">>> Summary saved to: {summary_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HY-Motion-1.0 Text-to-Motion Local Infer")
    parser.add_argument("--model_path", type=str, required=True, help="Configuration file path")
    parser.add_argument(
        "--device_ids", type=str, default=None, help="GPU device ID list, separated by commas, e.g.: 0,1,2,3"
    )
    parser.add_argument(
        "--prompt_engineering_model_path",
        type=str,
        default=None,
        help="Prompt engineering model path, for text rewriting and duration estimation",
    )
    parser.add_argument(
        "--prompt_engineering_host",
        type=str,
        default=None,
        help="Prompt engineering host address, for text rewriting and duration estimation",
    )
    parser.add_argument("--input_text_dir", type=str, default=None, help="Input text directory")
    parser.add_argument("--output_dir", type=str, default="output/local_infer", help="Output directory")
    parser.add_argument("--cfg_scale", type=float, default=5.0, help="CFG scale factor")
    parser.add_argument("--validation_steps", type=int, default=None, help="Validation steps")
    parser.add_argument("--disable_rewrite", action="store_true", help="Disable text rewriting")
    parser.add_argument("--disable_duration_est", action="store_true", help="Disable duration estimation")
    parser.add_argument("--num_seeds", type=int, default=4, help="Number of random seeds")
    
    parser.add_argument(
        "--ref_motion_path", type=str, default=None,
        help="Reference motion path (.npy/.npz/.pkl) for TM2M; omit for T2M",
    )
    parser.add_argument(
        "--ref_motion_fps", type=float, default=30.0,
        help="FPS of the reference motion, used for resampling to model fps",
    )

    args = parser.parse_args()

    # check required files
    if args.model_path.endswith(".pt"):
        cfg = os.path.join(os.path.dirname(args.model_path), "config.yaml")
        ckpt = args.model_path
    else:
        cfg = osp.join(args.model_path, "config.yml")
        ckpt = osp.join(args.model_path, "latest.ckpt")
    if not os.path.exists(cfg):
        raise FileNotFoundError(f">>> Configuration file not found: {cfg}")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f">>> Checkpoint file not found: {ckpt}")

    # parse device IDs
    device_ids = None
    if args.device_ids:
        try:
            device_ids = [int(x.strip()) for x in args.device_ids.split(",")]
            print(f">>> Specified GPU devices: {device_ids}")
        except ValueError:
            raise ValueError(f"Invalid GPU device ID: {args.device_ids}")

    # Initialize runtime
    print(">>> Initializing T2MRuntime...")
    runtime = T2MRuntime(
        config_path=cfg,
        ckpt_name=ckpt,
        device_ids=device_ids,
        disable_prompt_engineering=args.disable_duration_est and args.disable_rewrite,
        prompt_engineering_host=args.prompt_engineering_host,
        prompt_engineering_model_path=args.prompt_engineering_model_path,
    )

    # set validation steps
    if args.validation_steps is not None:
        for pipeline in runtime.pipelines:
            pipeline.validation_steps = args.validation_steps

    # determine input files
    if args.input_text_dir is None:
        input_text_files = ["examples/example_prompts/example_subset.json"]
    else:
        input_text_files = parse_dirs_and_sort(
            args.input_text_dir, suffix=".json", with_prefix=True
        ) + parse_dirs_and_sort(args.input_text_dir, suffix=".txt", with_prefix=True)

    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    ref_motion = None
    if args.ref_motion_path is not None:
        ref_motion = load_ref_motion(
            path=args.ref_motion_path,
            fps=args.ref_motion_fps,
            runtime=runtime,
        )

    # batch process files
    results_list = []
    for input_file in input_text_files:
        if not os.path.exists(input_file):
            print(f">>> Skipping non-existent file: {input_file}")
            continue
        result = process_input_file(
            runtime=runtime,
            input_file=input_file,
            output_dir=args.output_dir,
            cfg_scale=args.cfg_scale,
            disable_rewrite=args.disable_rewrite,
            disable_duration_est=args.disable_duration_est,
            num_seeds=args.num_seeds,
            ref_motion=ref_motion,
        )
        results_list.append(result)

    # save batch results
    save_batch_results(results_list, args.output_dir)

    print(">>> Batch processing completed!")


if __name__ == "__main__":
    """
    python local_infer.py --model_path ckpts/tencent/HY-Motion-1.0 \
        --device_ids 0,1 \
        --input_text_dir examples/example_prompts/ \
    """
    main()
