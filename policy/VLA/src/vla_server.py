"""Common HTTP contract for model-specific server-side VLA adapters."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from model_config import DEFAULT_CONFIG_PATH, load_model_profile, validate_checkpoint_revision
from repo_paths import resolve_paths
from reproducibility import build_reproducibility_manifest, require_reproducible_source


MAX_REQUEST_BYTES = 64 * 1024 * 1024


def import_symbol(path: str) -> Any:
    module_name, separator, symbol_name = path.partition(":")
    if not separator:
        raise ValueError(f"Expected '<module>:<symbol>', got {path!r}")
    return getattr(importlib.import_module(module_name), symbol_name)


def make_handler(adapter: Any) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "OSMA-Bench-VLA/1"

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _read_payload(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError(f"Request body size {length} is outside (0, {MAX_REQUEST_BYTES}]")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            if self.path == "/health":
                self._send(200, {"ok": True, "status": "ready"})
            elif self.path == "/meta":
                self._send(200, adapter.metadata())
            else:
                self._send(404, {"error": "not_found", "path": self.path})

        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            try:
                payload = self._read_payload()
                if self.path == "/reset":
                    result = adapter.reset(payload)
                elif self.path == "/act":
                    result = adapter.act(payload)
                else:
                    self._send(404, {"error": "not_found", "path": self.path})
                    return
                self._send(200, result)
            except Exception as error:  # keep server alive and return an auditable error
                logging.exception("Request %s failed", self.path)
                self._send(
                    500,
                    {
                        "error": type(error).__name__,
                        "detail": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )

        def log_message(self, fmt: str, *args: Any) -> None:
            logging.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a VLA model behind the OSMA-Bench HTTP contract")
    parser.add_argument("--model-profile", default="xr1")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path", default=None, help="HF repo id or local checkpoint directory")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Immutable checkpoint commit SHA; default comes from the model profile",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-implementation", default="flash_attention_2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--require-reproducible-source",
        action="store_true",
        help="Refuse startup unless relevant code is exactly recoverable from the recorded Git commit",
    )
    parser.add_argument("--print-profile", action="store_true", help="Print validated config without loading weights")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    profile = load_model_profile(args.model_profile, args.config)
    if args.print_profile:
        print(json.dumps(profile, indent=2))
        return

    model_path = args.model_path or profile["checkpoint"]
    model_revision = validate_checkpoint_revision(
        args.model_revision or profile["checkpoint_revision"]
    )
    reproducibility = build_reproducibility_manifest(
        resolve_paths().superproject_root,
        resolved_args={**vars(args), "model_path": model_path, "model_revision": model_revision},
        package_names=(
            "flash_attn",
            "huggingface-hub",
            "numpy",
            "opentau",
            "Pillow",
            "safetensors",
            "tokenizers",
            "torch",
            "transformers",
        ),
    )
    if args.require_reproducible_source:
        require_reproducible_source(reproducibility)
    adapter_class = import_symbol(profile["adapter"])
    logging.info(
        "Loading %s from %s at revision %s on %s",
        args.model_profile,
        model_path,
        model_revision,
        args.device,
    )
    adapter = adapter_class(
        profile=profile,
        model_path=model_path,
        model_revision=model_revision,
        device=args.device,
        attention_implementation=args.attention_implementation,
        reproducibility=reproducibility,
    )
    # One serial request stream per GPU/model. The model is not assumed to be
    # thread-safe, and this first milestone has exactly one simulator client.
    server = HTTPServer((args.host, args.port), make_handler(adapter))
    logging.info("VLA server ready at http://%s:%d", args.host, args.port)
    logging.info("Model metadata: %s", json.dumps(adapter.metadata(), sort_keys=True))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping VLA server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
