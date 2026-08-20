#!/usr/bin/env python3
"""Stream-merge STEAD noise and earthquake chunks for binary detection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py


def trace_count(path: Path) -> int:
    with h5py.File(path, "r") as h5:
        if "data" not in h5:
            raise ValueError(f"{path} does not contain a data group.")
        return len(h5["data"])


def merge_hdf5(inputs: list[Path], output: Path) -> int:
    """Copy one trace dataset at a time; never materialize all waveforms."""
    expected = sum(trace_count(path) for path in inputs)
    copied = 0

    with h5py.File(output, "w") as destination:
        destination_group = destination.create_group("data")
        for input_path in inputs:
            with h5py.File(input_path, "r") as source:
                source_group = source["data"]
                for trace_name in source_group:
                    if trace_name in destination_group:
                        raise ValueError(f"Duplicate trace name: {trace_name}")
                    # h5py performs the transfer directly between files.
                    source.copy(source_group[trace_name], destination_group, name=trace_name)
                    copied += 1
                    if copied % 10_000 == 0:
                        print(f"Copied {copied:,}/{expected:,} traces")

    actual = trace_count(output)
    if actual != expected or copied != expected:
        raise RuntimeError(f"Verification failed: expected {expected:,}, copied {copied:,}, output has {actual:,}.")
    print(f"HDF5 verified: {actual:,} traces in {output}")
    return actual


def merge_csv(inputs: list[Path], output: Path, expected_rows: int) -> None:
    """Stream metadata rows, retaining each original trace_name."""
    written = 0
    fieldnames: list[str] | None = None
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = None
        for input_path in inputs:
            with input_path.open("r", newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                    if not fieldnames or "trace_name" not in fieldnames:
                        raise ValueError(f"{input_path} is missing trace_name.")
                    writer = csv.DictWriter(destination, fieldnames=fieldnames)
                    writer.writeheader()
                elif reader.fieldnames != fieldnames:
                    raise ValueError(f"CSV columns do not match: {input_path}")
                for row in reader:
                    writer.writerow(row)
                    written += 1
    if written != expected_rows:
        raise RuntimeError(f"CSV verification failed: expected {expected_rows:,}, wrote {written:,} rows.")
    print(f"CSV verified: {written:,} rows in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two STEAD chunks for detection training.")
    parser.add_argument("--hdf5", nargs=2, required=True, metavar=("CHUNK1", "CHUNK2"))
    parser.add_argument("--csv", nargs=2, required=True, metavar=("CHUNK1", "CHUNK2"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_inputs = [Path(path) for path in args.hdf5]
    csv_inputs = [Path(path) for path in args.csv]
    for path in hdf5_inputs + csv_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    hdf5_output = output_dir / "detection.hdf5"
    csv_output = output_dir / "detection.csv"
    if hdf5_output.exists() or csv_output.exists():
        raise FileExistsError("detection.hdf5 or detection.csv already exists; refuse to overwrite it.")

    expected = merge_hdf5(hdf5_inputs, hdf5_output)
    merge_csv(csv_inputs, csv_output, expected)
    print("Merge complete.")


if __name__ == "__main__":
    main()
