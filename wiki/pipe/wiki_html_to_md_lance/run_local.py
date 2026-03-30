#!/usr/bin/env python3
"""Run wiki_html_to_md_lance pipe locally on a Lance dataset."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import lance
import pyarrow as pa


def _bootstrap_paths() -> Path:
    """Ensure repo root and local DCD repos are importable."""
    root = Path(__file__).resolve().parents[3]
    dcd_root = root / "refer_repo" / "dcd"
    dcd_cli_root = root / "refer_repo" / "dcd-cli"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(dcd_root) not in sys.path:
        sys.path.insert(0, str(dcd_root))
    if str(dcd_cli_root) not in sys.path:
        sys.path.insert(0, str(dcd_cli_root))
    return root


ROOT = _bootstrap_paths()

from dcd_cli.pipe import PipeContext  # noqa: E402
from dataclawdev.data.util.prepare_dataset import run as prepare_dataset  # noqa: E402
from wiki.pipe.wiki_html_to_md_lance import map as pipe_map  # noqa: E402

_WORKER_MAP = None
_WORKER_CTX = None


def _build_table(out_batch: dict[str, list[Any]], schema: pa.Schema) -> pa.Table:
    arrays: dict[str, pa.Array] = {}
    for field in schema:
        values = out_batch.get(field.name)
        if values is None:
            raise KeyError(f"Missing output column: {field.name}")
        arrays[field.name] = pa.array(values, type=field.type)
    return pa.table(arrays, schema=schema)


def _link_or_replace(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    rel = os.path.relpath(src, dst.parent)
    os.symlink(rel, dst)


def _worker_init(
    root_str: str,
    dataset: str,
    dataset_dir_str: str,
    remove_ref: bool,
    max_item_seconds: int,
    rewrite_images: bool,
) -> None:
    global _WORKER_MAP, _WORKER_CTX
    root = Path(root_str)
    dcd_root = root / "refer_repo" / "dcd"
    dcd_cli_root = root / "refer_repo" / "dcd-cli"
    for p in (str(root), str(dcd_root), str(dcd_cli_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from dcd_cli.pipe import PipeContext as _PipeContext
    from wiki.pipe.wiki_html_to_md_lance import map as _pipe_map

    _WORKER_MAP = _pipe_map
    _WORKER_CTX = _PipeContext(
        dataset=dataset,
        pipe_name="wiki_html_to_md_lance",
        pipe_version=1,
        dataset_dir=Path(dataset_dir_str),
        config={
            "remove_ref": remove_ref,
            "max_item_seconds": max_item_seconds,
            "rewrite_images": rewrite_images,
            "dataset_dir": dataset_dir_str,
        },
    )


def _worker_process(
    payload: tuple[int, dict[str, list[Any]]],
) -> tuple[int, int, dict[str, list[Any]]]:
    idx, batch = payload
    out_batch = _WORKER_MAP(batch, _WORKER_CTX)
    row_count = len(next(iter(out_batch.values()))) if out_batch else 0
    return idx, row_count, out_batch


def _iter_batches(
    ds: Any,
    batch_size: int,
) -> Iterator[tuple[int, dict[str, list[Any]]]]:
    scanner = ds.scanner(batch_size=batch_size)
    for idx, rb in enumerate(scanner.to_batches(), start=1):
        yield idx, {name: rb.column(name).to_pylist() for name in rb.schema.names}


def _verify_same_order(src_text: Path, dst_text: Path) -> None:
    """Ensure output text.lance keeps the same row order as input."""
    src_ids = (
        lance.dataset(str(src_text))
        .to_table(columns=["id"])
        .column("id")
        .to_pylist()
    )
    dst_ids = (
        lance.dataset(str(dst_text))
        .to_table(columns=["id"])
        .column("id")
        .to_pylist()
    )
    if len(src_ids) != len(dst_ids):
        raise RuntimeError(
            f"Row count mismatch after conversion: src={len(src_ids)} dst={len(dst_ids)}",
        )
    for idx, (sid, did) in enumerate(zip(src_ids, dst_ids, strict=True)):
        if sid != did:
            raise RuntimeError(
                f"Row order mismatch at index={idx}: src_id={sid}, dst_id={did}",
            )


def run(
    src_dataset_dir: Path,
    dst_dataset_dir: Path,
    *,
    batch_size: int = 64,
    remove_ref: bool = False,
    max_item_seconds: int = 8,
    rewrite_images: bool = True,
    workers: int = 1,
    run_prepare: bool = True,
) -> None:
    src_text = src_dataset_dir / "text.lance"
    if not src_text.is_dir():
        raise FileNotFoundError(f"Source text.lance not found: {src_text}")

    dst_dataset_dir.mkdir(parents=True, exist_ok=True)
    dst_text = dst_dataset_dir / "text.lance"
    if dst_text.exists():
        shutil.rmtree(dst_text)

    ds = lance.dataset(str(src_text))
    schema = ds.schema
    total = ds.count_rows()

    ctx = PipeContext(
        dataset=src_dataset_dir.name,
        pipe_name="wiki_html_to_md_lance",
        pipe_version=1,
        dataset_dir=src_dataset_dir,
        config={
            "remove_ref": remove_ref,
            "max_item_seconds": max_item_seconds,
            "rewrite_images": rewrite_images,
            "dataset_dir": str(src_dataset_dir),
        },
    )

    print(f"source: {src_dataset_dir}")
    print(f"target: {dst_dataset_dir}")
    print(
        f"rows: {total}, batch_size: {batch_size}, remove_ref={remove_ref}, "
        f"max_item_seconds={max_item_seconds}, rewrite_images={rewrite_images}",
    )

    t0 = time.time()
    done = 0
    first_batch_written = False
    if workers <= 1:
        result_iter = (
            (idx, len(next(iter(batch.values()))) if batch else 0, pipe_map(batch, ctx))
            for idx, batch in _iter_batches(ds, batch_size)
        )
        for i, row_count, out_batch in result_iter:
            out_table = _build_table(out_batch, schema)
            mode = "overwrite" if not first_batch_written else "append"
            lance.write_dataset(out_table, str(dst_text), mode=mode)
            first_batch_written = True

            done += row_count
            if i % 20 == 0 or done == total:
                elapsed = time.time() - t0
                eta = (total - done) * elapsed / done if done else 0.0
                print(
                    f"progress: {done}/{total} ({done * 100 / total:.1f}%) | "
                    f"elapsed {elapsed:.1f}s | eta {eta:.1f}s",
                )
    else:
        mp_ctx = mp.get_context("spawn")
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_ctx,
            initializer=_worker_init,
            initargs=(
                str(ROOT),
                src_dataset_dir.name,
                str(src_dataset_dir),
                remove_ref,
                max_item_seconds,
                rewrite_images,
            ),
        ) as pool:
            pending: dict[cf.Future[tuple[int, int, dict[str, list[Any]]]], int] = {}
            buffered: dict[int, tuple[int, dict[str, list[Any]]]] = {}
            next_write_idx = 1
            batch_iter = iter(_iter_batches(ds, batch_size))
            max_pending = max(1, workers * 4)

            def _submit_until_full() -> None:
                while len(pending) < max_pending:
                    try:
                        payload = next(batch_iter)
                    except StopIteration:
                        return
                    fut = pool.submit(_worker_process, payload)
                    pending[fut] = payload[0]

            _submit_until_full()
            while pending:
                done_set, _ = cf.wait(
                    pending.keys(),
                    return_when=cf.FIRST_COMPLETED,
                )
                for fut in done_set:
                    pending.pop(fut, None)
                    idx, row_count, out_batch = fut.result()
                    buffered[idx] = (row_count, out_batch)

                while next_write_idx in buffered:
                    row_count, out_batch = buffered.pop(next_write_idx)
                    out_table = _build_table(out_batch, schema)
                    mode = "overwrite" if not first_batch_written else "append"
                    lance.write_dataset(out_table, str(dst_text), mode=mode)
                    first_batch_written = True

                    done += row_count
                    if next_write_idx % 20 == 0 or done == total:
                        elapsed = time.time() - t0
                        eta = (total - done) * elapsed / done if done else 0.0
                        print(
                            f"progress: {done}/{total} ({done * 100 / total:.1f}%) | "
                            f"elapsed {elapsed:.1f}s | eta {eta:.1f}s",
                        )
                    next_write_idx += 1

                _submit_until_full()

    _verify_same_order(src_text, dst_text)

    for table_name in ("images.lance", "image_labels.lance"):
        src_table = src_dataset_dir / table_name
        dst_table = dst_dataset_dir / table_name
        if src_table.is_dir():
            _link_or_replace(src_table, dst_table)

    if run_prepare:
        print("prepare_dataset: start (tokenizer=simple)")
        prepare_dataset(dst_dataset_dir, base_tokenizer="simple")
        print("prepare_dataset: done")

    print(f"done in {time.time() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wiki_html_to_md_lance pipe locally")
    parser.add_argument(
        "src_dataset_dir",
        nargs="?",
        type=Path,
        default=ROOT / "workspace" / "html_lance" / "wiki_0320_en_has_pic",
    )
    parser.add_argument(
        "dst_dataset_dir",
        nargs="?",
        type=Path,
        default=ROOT / "workspace" / "md_lance" / "wiki_0320_en_has_pic",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--remove-ref", action="store_true")
    parser.add_argument("--max-item-seconds", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--no-rewrite-images", action="store_true")
    parser.add_argument("--no-prepare", action="store_true")
    args = parser.parse_args()

    run(
        args.src_dataset_dir.resolve(),
        args.dst_dataset_dir.resolve(),
        batch_size=max(1, args.batch_size),
        remove_ref=args.remove_ref,
        max_item_seconds=args.max_item_seconds,
        rewrite_images=not args.no_rewrite_images,
        workers=max(1, args.workers),
        run_prepare=not args.no_prepare,
    )


if __name__ == "__main__":
    main()
