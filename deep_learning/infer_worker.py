"""Persistent ONNX Runtime worker for MATLAB perception loops."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASSES = ["background", "drydown", "weed_cluster", "double_plant"]


def preprocess(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor, (h, w)


def run_request(session: ort.InferenceSession, input_name: str, request: dict[str, object]) -> dict[str, object]:
    image_path = Path(str(request["image"]))
    output_mask = Path(str(request["output_mask"]))
    image_size = int(request["image_size"])
    start = time.perf_counter()

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    tensor, original_hw = preprocess(image, image_size)
    logits = session.run(None, {input_name: tensor})[0]
    mask = logits.argmax(axis=1)[0].astype(np.uint8)
    mask = cv2.resize(mask, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_NEAREST)

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask), mask)
    output_mask.with_suffix(".json").write_text(
        json.dumps(
            {
                "image": str(image_path),
                "mask": str(output_mask),
                "classes": CLASSES,
                "image_size": image_size,
                "worker_inference_time_ms": (time.perf_counter() - start) * 1000.0,
            },
            indent=2,
        )
    )
    return {"ok": True, "message": "ok", "elapsed_ms": (time.perf_counter() - start) * 1000.0}


def write_json(path: Path, data: dict[str, object]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def read_request_when_ready(path: Path) -> dict[str, object] | None:
    """Read a MATLAB request file after Windows/OneDrive releases its lock."""
    for _ in range(200):
        try:
            text = path.read_text()
            if not text.strip():
                time.sleep(0.01)
                continue
            return json.loads(text)
        except (PermissionError, json.JSONDecodeError, OSError):
            time.sleep(0.01)
    return None


def main(args: argparse.Namespace) -> None:
    args.request_dir.mkdir(parents=True, exist_ok=True)
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_name = session.get_inputs()[0].name
    write_json(args.ready_file, {"ok": True, "model": str(args.model), "providers": providers})

    with args.log_file.open("a", encoding="utf-8") as log:
        log.write(f"ready model={args.model}\n")
        log.flush()
        while not args.stop_file.exists():
            requests = sorted(args.request_dir.glob("request_*.json"))
            if not requests:
                time.sleep(0.01)
                continue
            for request_path in requests:
                request = read_request_when_ready(request_path)
                if request is None:
                    time.sleep(0.05)
                    continue
                try:
                    response_path = Path(str(request["response"]))
                    response = run_request(session, input_name, request)
                    write_json(response_path, response)
                    request_path.unlink(missing_ok=True)
                except Exception as exc:  # MATLAB reads this message.
                    response_path = None
                    try:
                        request = json.loads(request_path.read_text())
                        response_path = Path(str(request.get("response", "")))
                    except Exception:
                        pass
                    log.write(f"error request={request_path} message={exc}\n")
                    log.flush()
                    if response_path:
                        write_json(response_path, {"ok": False, "message": str(exc)})
                    request_path.unlink(missing_ok=True)
        log.write("stopped\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
