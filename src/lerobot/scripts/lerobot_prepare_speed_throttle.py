#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Prepare strict speed-adapter throttle JSONL from trace v2 and human labels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from lerobot.rollout.speed_adapter_data import (
    annotation_jsonl_schema,
    annotation_template_rows,
    convert_trace_annotations,
    load_realtime_trace_v2,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="RealtimeTraceWriter schema v2 JSONL")
    parser.add_argument("--annotations", type=Path, help="Completed human annotation JSONL")
    parser.add_argument("--output", type=Path, help="Destination lerobot.speed_throttle.v1 JSONL")
    parser.add_argument(
        "--export-annotation-template",
        type=Path,
        help="Write one deliberately unlabeled template row per trace chunk",
    )
    parser.add_argument("--print-annotation-schema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_annotation_schema:
        print(json.dumps(annotation_jsonl_schema(), indent=2, sort_keys=True))
        return 0
    try:
        if args.export_annotation_template is not None:
            if args.trace is None:
                raise ValueError("--trace is required with --export-annotation-template")
            if args.annotations is not None or args.output is not None:
                raise ValueError("template export cannot be combined with --annotations or --output")
            chunks = load_realtime_trace_v2(args.trace)
            write_jsonl(
                args.export_annotation_template,
                annotation_template_rows(chunks),
                overwrite=args.overwrite,
            )
            print(
                f"wrote {len(chunks)} unlabeled annotation template rows to {args.export_annotation_template}"
            )
            return 0
        if args.trace is None or args.annotations is None or args.output is None:
            raise ValueError("--trace, --annotations, and --output are required for conversion")
        provenance = convert_trace_annotations(
            args.trace,
            args.annotations,
            args.output,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"wrote {provenance['num_segments']} labeled segments from "
        f"{provenance['num_chunks']} chunks to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
