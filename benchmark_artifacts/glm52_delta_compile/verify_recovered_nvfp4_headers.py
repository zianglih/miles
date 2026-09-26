#!/usr/bin/env python3
"""Compare new conversion headers with retained original CPU-replay evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    evidence = root / "artifacts/recovery-original-canonical-headers.json"
    expected = json.loads(evidence.read_text())
    checkpoint = root / "models/GLM-5.2_5layer-NVFP4"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values())
    if names != {row["name"] for row in expected["canonical_files"]}:
        raise ValueError("Converted shard set differs from original canonical checkpoint")
    actual, tensor_count = [], 0
    for row in expected["canonical_files"]:
        path = checkpoint / row["name"]
        with path.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"Truncated header: {path}")
            header_size = struct.unpack("<Q", size_bytes)[0]
            if header_size > path.stat().st_size - 8:
                raise ValueError(f"Truncated file: {path}")
            header = stream.read(header_size)
        record = {"name": row["name"], "file_bytes": path.stat().st_size,
                  "header_sha256": hashlib.sha256(header).hexdigest()}
        if record != row:
            raise ValueError(f"Recovered canonical header/size differs: {record}")
        tensors = set(json.loads(header)) - {"__metadata__"}
        if tensors != {name for name, shard in index["weight_map"].items() if shard == row["name"]}:
            raise ValueError(f"Recovered index/header mapping differs: {row['name']}")
        tensor_count += len(tensors)
        actual.append(record)
    if tensor_count != expected["canonical_tensor_count"]:
        raise ValueError("Canonical tensor count differs")
    report = {"matched_original_headers": True, "canonical_tensor_count": tensor_count,
              "files": actual, "expected_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
              "original_provenance_sha256": expected["provenance_sha256"], "limit": expected["limit"]}
    output = root / "artifacts/recovery-nvfp4-header-verification.json"
    text = json.dumps(report, indent=2) + "\n"
    if output.exists() and output.read_text() != text:
        raise ValueError(f"Do not overwrite different retained verification: {output}")
    if not output.exists():
        output.write_text(text)
    print(json.dumps({"matched_original_headers": True, "shards": len(actual), "tensors": tensor_count,
                      "report": str(output), "report_sha256": hashlib.sha256(text.encode()).hexdigest()}))


if __name__ == "__main__":
    main()
