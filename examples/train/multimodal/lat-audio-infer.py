import os
import json
import uuid
import argparse
import subprocess
from typing import Dict, Any, List, Set

import torch
from swift.infer_engine import InferRequest, TransformersEngine, RequestConfig
from swift.agent_template import agent_template_map


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


TASK_TOOL_JSON: Dict[str, str] = {
    "DAC_CN": """[{"type": "function", "function": {"name": "crop_audio", "description": "裁剪音频片段进行细节观察", "parameters": {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}}, "required": ["start_sec", "end_sec"]}}}]""",
    "DAC_EN": """[{"type": "function", "function": {"name": "crop_audio", "description": "Crop audio segments for detailed observation", "parameters": {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}}, "required": ["start_sec", "end_sec"]}}}]""",
    "TAC_CN": """[{"type": "function", "function": {"name": "crop_audio", "description": "裁剪片段", "parameters": {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}}, "required": ["start_sec", "end_sec"]}}}]""",
    "TAC_EN": """[{"type": "function", "function": {"name": "crop_audio", "description": "crop audio", "parameters": {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}}, "required": ["start_sec", "end_sec"]}}}]""",
    "TAG_CN": """[{"type": "function", "function": {"name": "crop_audio", "description": "裁剪音频", "parameters": {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}}, "required": ["start_sec", "end_sec"]}}}]""",
    "TAG_EN": """[{"type": "function", "function": {"name": "crop_audio", "description": "crop audio", "parameters": {"type": "object", "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}}, "required": ["start_sec", "end_sec"]}}}]""",
}


def ensure_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [ensure_serializable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return ensure_serializable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return ensure_serializable(obj.__dict__)
    return obj


def physical_crop(src_path: str, start: float, end: float, crop_dir: str) -> str:
    os.makedirs(crop_dir, exist_ok=True)
    unique_fn = f"crop_{uuid.uuid4().hex[:8]}.mp3"
    target_path = os.path.join(crop_dir, unique_fn)
    duration = max(0.1, end - start)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-t",
        str(duration),
        "-i",
        src_path,
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        target_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return target_path


def load_done_indices(output_path: str) -> Set[int]:
    done = set()
    if not os.path.exists(output_path):
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sample_index = obj.get("sample_index")
                if sample_index is not None:
                    done.add(int(sample_index))
            except Exception:
                continue
    return done


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def run_task_infer(
    ckpt_path: str,
    task: str,
    test_file: str,
    crop_dir: str,
    output_path: str,
    error_path: str,
    agent_template_name: str = "hermes",
    max_rounds: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    resume: bool = True,
):
    if task not in TASK_TOOL_JSON:
        raise ValueError(f"Unknown task: {task}")

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    tools = json.loads(TASK_TOOL_JSON[task])

    os.makedirs(crop_dir, exist_ok=True)
    output_dir = os.path.dirname(output_path)
    error_dir = os.path.dirname(error_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if error_dir:
        os.makedirs(error_dir, exist_ok=True)

    done_indices = load_done_indices(output_path) if resume else set()

    print(f"Loading model from: {ckpt_path}")
    engine = TransformersEngine(ckpt_path)
    engine.template.agent_template = agent_template_map[agent_template_name]()

    stop_word = engine.template.agent_template.keyword.observation
    request_config = RequestConfig(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=[stop_word],
    )

    with open(test_file, "r", encoding="utf-8") as f:
        test_lines = f.readlines()

    print(f"Task: {task}")
    print(f"Test file: {test_file}")
    print(f"Total samples: {len(test_lines)}")
    print(f"Already finished: {len(done_indices)}")
    print("=" * 80)

    for i, line in enumerate(test_lines):
        sample_index = i + 1

        if sample_index in done_indices:
            print(f"[Skip] sample {sample_index} already finished.")
            continue

        try:
            data = json.loads(line)
            audio_path = data["audios"][0]
            query = data["messages"][0]["content"]
            ground_truth = data["messages"][1]["content"]

            print(f"\n[Sample {sample_index}/{len(test_lines)}] {task}")
            print(f"[Query] {query}")
            print(f"[GT] {ground_truth}")
            print("-" * 40)

            infer_request = InferRequest(
                messages=[{"role": "user", "content": query}],
                tools=tools,
                audios=[audio_path],
            )

            final_prediction = ""
            created_crops: List[str] = []

            for r in range(max_rounds):
                resp_list = engine.infer([infer_request], request_config)
                response_msg = resp_list[0].choices[0].message

                if response_msg.content:
                    print(f"\n[Round {r + 1}]\n{response_msg.content}")

                if response_msg.tool_calls:
                    formatted_tcs = ensure_serializable(response_msg.tool_calls)

                    for tc in response_msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        start = args.get("start_sec", 0)
                        end = args.get("end_sec", 0)
                        print(f"[Tool] crop_audio({start}, {end})")

                        crop_file = physical_crop(audio_path, start, end, crop_dir)
                        created_crops.append(crop_file)

                        tool_res_str = json.dumps({"result": "Segment extracted: <audio>"}, ensure_ascii=False)

                        infer_request.messages.append({
                            "role": "assistant",
                            "content": response_msg.content,
                            "tool_calls": formatted_tcs,
                        })
                        infer_request.messages.append({
                            "role": "tool",
                            "content": tool_res_str,
                            "tool_call_id": tc.id,
                        })
                        infer_request.audios.append(crop_file)

                    print("[Info] Cropped audio appended. Continue reasoning...")
                else:
                    final_prediction = response_msg.content or ""
                    print(f"\n[Final Prediction] {final_prediction}")
                    infer_request.messages.append({
                        "role": "assistant",
                        "content": final_prediction,
                    })
                    break

            output_entry = {
                "sample_index": sample_index,
                "task": task,
                "audio_path": audio_path,
                "query": query,
                "ground_truth": ground_truth,
                "final_prediction": final_prediction,
                "trajectory": ensure_serializable(infer_request.messages),
                "all_audio_paths": infer_request.audios,
            }

            append_jsonl(output_path, output_entry)

            # Optional cleanup of temporary crops
            for fp in created_crops:
                if os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except Exception:
                        pass

            print("=" * 80)

        except Exception as e:
            err_entry = {
                "sample_index": sample_index,
                "task": task,
                "error": str(e),
            }
            append_jsonl(error_path, err_entry)
            print(f"[Error] sample {sample_index}: {e}")
            continue

    print(f"\nFinished task: {task}")
    print(f"Output file: {output_path}")
    print(f"Error file: {error_path}")


def main():
    parser = argparse.ArgumentParser(description="LAT-Audio inference with tool-augmented reasoning")

    parser.add_argument("--ckpt", type=str, required=True, help="Local path to the model or checkpoint")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_TOOL_JSON.keys()),
        help="Task type",
    )
    parser.add_argument("--test_file", type=str, required=True, help="Path to the JSONL evaluation file")
    parser.add_argument("--crop_dir", type=str, default="./tmp_crops", help="Directory for temporary cropped audio")
    parser.add_argument("--output", type=str, default="./results/infer_result.jsonl", help="Output JSONL path")
    parser.add_argument("--error_output", type=str, default="./results/infer_error.jsonl", help="Error JSONL path")
    parser.add_argument("--template", type=str, default="hermes", help="Agent template name")
    parser.add_argument("--rounds", type=int, default=6, help="Maximum number of reasoning rounds")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum generation tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume mode")

    args = parser.parse_args()

    run_task_infer(
        ckpt_path=args.ckpt,
        task=args.task,
        test_file=args.test_file,
        crop_dir=args.crop_dir,
        output_path=args.output,
        error_path=args.error_output,
        agent_template_name=args.template,
        max_rounds=args.rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
