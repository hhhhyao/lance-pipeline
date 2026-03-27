#!/usr/bin/env python3
"""Convert wiki part*.jsonl + part*.tar into Lance datasets.

Output tables:
- text.lance
- images.lance
- image_labels.lance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import tarfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import lance
import pyarrow as pa

log = logging.getLogger(__name__)

DEFAULT_LOG_INTERVAL = 250

TEXT_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("info", pa.string()),
    pa.field("data", pa.large_string()),
    pa.field("tags", pa.list_(pa.string())),
])

IMAGES_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("image_bytes", pa.binary()),
    pa.field("sha256", pa.string()),
])

IMAGE_LABELS_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("info", pa.string()),
    pa.field("data", pa.string()),
    pa.field("tags", pa.list_(pa.string())),
])

TITLE_RE = re.compile(r"<title>([^<]+)</title>", re.IGNORECASE)
CANONICAL_RE = re.compile(
    r'<link[^>]*rel=["\']canonical["\'][^>]*href=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
CATEGORY_RE = re.compile(
    r'<a[^>]*href=["\'](?:https?://[^/]*)*/wiki/Category:[^"\']+["\'][^>]*>'
    r"([^<]+)</a>",
    re.IGNORECASE,
)

THUMB_SIZE_RE = re.compile(r"/thumb/(.*)/\d+px-[^/]+$")
IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
IMG_SRC_RE = re.compile(r'(\bsrc=["\'])([^"\']+)(["\'])', re.IGNORECASE)
SRCSET_RE = re.compile(r'\s*\bsrcset=["\'][^"\']*["\']', re.IGNORECASE)
WIKIMEDIA_RE = re.compile(r"upload\.wikimedia\.org", re.IGNORECASE)


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def normalize_url(url: str) -> str:
    """Strip protocol and normalize Wikimedia thumbnail URLs."""
    url = unquote(re.sub(r"^https?:", "", url).lstrip("/"))
    m = THUMB_SIZE_RE.search(url)
    if m:
        url = url[: m.start()] + "/" + m.group(1)
    return url


def extract_html_meta(html: str) -> dict[str, object]:
    """Extract URL, title and category tags from HTML."""
    result: dict[str, object] = {}
    m = CANONICAL_RE.search(html)
    if m:
        result["url"] = m.group(1)

    m = TITLE_RE.search(html)
    if m:
        raw_title = m.group(1).strip()
        result["title"] = re.sub(r"\s*[-–—]\s*Wikipedia\s*$", "", raw_title)

    skip_prefixes = (
        "Category", "Articles", "Pages", "Webarchive", "CS1", "All ",
        "Use ", "Commons category", "Short description", "Wikipedia",
    )
    tags: list[str] = []
    for cm in CATEGORY_RE.finditer(html):
        cat = cm.group(1).strip()
        if not any(cat.startswith(s) for s in skip_prefixes):
            tags.append(cat.replace("&amp;", "&"))
    result["tags"] = tags
    return result


def rewrite_html_images(
    html: str,
    img_url_to_id: dict[str, str],
    available_ids: set[str],
) -> str:
    """Rewrite Wikimedia image URLs to local images/{id} when possible."""

    def replacer(m: re.Match[str]) -> str:
        tag = m.group(0)
        src_m = IMG_SRC_RE.search(tag)
        if not src_m:
            return tag
        src = src_m.group(2)
        if not WIKIMEDIA_RE.search(src):
            return tag

        image_id = img_url_to_id.get(normalize_url(src))
        if not image_id or image_id not in available_ids:
            return tag

        tag = SRCSET_RE.sub("", tag)
        return IMG_SRC_RE.sub(rf"\g<1>images/{image_id}\g<3>", tag)

    return IMG_TAG_RE.sub(replacer, html)


def _find_pairs(src_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for jf in sorted(src_dir.glob("*.jsonl")):
        tf = src_dir / f"{jf.stem}.tar"
        if tf.is_file():
            pairs.append((jf, tf))
    if not pairs:
        raise FileNotFoundError(f"No matched *.jsonl + *.tar pairs under {src_dir}")
    return pairs


def _count_non_empty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_part_images(tar_path: Path) -> dict[str, bytes]:
    """Load one part tar into memory keyed as part/name and plain name."""
    store: dict[str, bytes] = {}
    part_name = tar_path.stem
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            member_name = member.name.removeprefix("images/")
            fobj = tf.extractfile(member)
            if not fobj:
                continue
            raw = fobj.read()
            store[f"{part_name}/{member_name}"] = raw
            store[member_name] = raw
    return store


def _resolve_image_bytes(
    image_file: str,
    part_name: str,
    image_store: dict[str, bytes],
) -> tuple[str, bytes] | None:
    normalized = image_file.removeprefix("images/")
    candidates = (
        image_file,
        normalized,
        f"{part_name}/{normalized}",
    )
    for cid in candidates:
        raw = image_store.get(cid)
        if raw is not None:
            return cid, raw
    return None


def compact_lance(lance_path: Path, table_name: str) -> None:
    log.info("Compacting %s (%s) ...", lance_path, table_name)
    t0 = time.perf_counter()
    ds: Any = lance.dataset(str(lance_path))
    ds.optimize.compact_files()

    existing = {idx["name"] for idx in ds.list_indices()}
    plan: list[tuple[str, str]] = [("id", "BTREE")]
    if "tags" in ds.schema.names:
        plan.append(("tags", "LABEL_LIST"))
    for col, idx_type in plan:
        idx_name = f"{col}_idx"
        if idx_name not in existing:
            log.info("Creating %s index on '%s' ...", idx_type, col)
            ds.create_scalar_index(col, index_type=idx_type)

    ds.optimize.optimize_indices()
    stats = ds.cleanup_old_versions(
        older_than=timedelta(seconds=0),
        delete_unverified=True,
    )
    if stats.bytes_removed:
        log.info(
            "Cleaned up %d old versions, freed %.2f GB",
            stats.old_versions,
            stats.bytes_removed / 1e9,
        )
    log.info("Compact %s done in %s", table_name, _fmt_seconds(time.perf_counter() - t0))


def run_streaming(src_dir: Path, dst_dir: Path, *, log_interval: int) -> None:
    pairs = _find_pairs(src_dir)
    line_counts = [_count_non_empty_lines(jf) for jf, _ in pairs]
    total_articles = sum(line_counts)
    id_width = max(5, len(str(max(total_articles - 1, 0))))

    dst_dir.mkdir(parents=True, exist_ok=True)
    text_lance = dst_dir / "text.lance"
    images_lance = dst_dir / "images.lance"
    image_labels_lance = dst_dir / "image_labels.lance"

    seen_img_ids: set[str] = set()
    text_written = 0
    images_written = 0
    t0 = time.perf_counter()

    log.info(
        "Start: %d parts, %d articles total, id_width=%d, log_interval=%d",
        len(pairs), total_articles, id_width, log_interval,
    )

    for part_idx, ((jf, tf), part_total) in enumerate(zip(pairs, line_counts), start=1):
        part_name = tf.stem
        part_t0 = time.perf_counter()
        image_store = _load_part_images(tf)
        available_ids = set(image_store.keys())

        log.info(
            "Part %d/%d: %s (%d lines), loaded %d tar images in %s",
            part_idx, len(pairs), jf.name, part_total, len(image_store),
            _fmt_seconds(time.perf_counter() - part_t0),
        )

        text_ids: list[str] = []
        text_infos: list[str] = []
        text_datas: list[str] = []
        text_tags: list[list[str]] = []

        img_ids: list[str] = []
        img_bytes_list: list[bytes] = []
        img_sha256_list: list[str] = []

        imgdata_ids: list[str] = []
        imgdata_infos: list[str] = []
        imgdata_datas: list[str] = []
        imgdata_tags: list[list[str]] = []

        processed_in_part = 0
        with jf.open("r", encoding="utf-8") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                processed_in_part += 1
                entry = json.loads(raw_line)

                html_data = str(entry.get("html", ""))
                meta = extract_html_meta(html_data)
                url = str(entry.get("final_url", meta.get("url", "")))
                title = str(meta.get("title", ""))
                raw_tags = meta.get("tags", [])
                tags = list(raw_tags) if isinstance(raw_tags, list) else []

                article_id = str(text_written + processed_in_part - 1).zfill(id_width)
                article_images = entry.get("images", [])
                image_ids_for_article: list[str] = []

                # Used only when rewrite_html_images() is enabled:
                # img_url_to_id: dict[str, str] = {}

                for img_meta in article_images:
                    image_file = str(img_meta.get("image_file", ""))
                    if not image_file:
                        continue
                    resolved = _resolve_image_bytes(image_file, part_name, image_store)
                    if not resolved:
                        continue
                    image_id, raw_bytes = resolved

                    # Used only when rewrite_html_images() is enabled:
                    # image_url = str(img_meta.get("image_url", ""))
                    # if image_url:
                    #     img_url_to_id[normalize_url(image_url)] = image_id

                    image_ids_for_article.append(image_id)
                    if image_id in seen_img_ids:
                        continue
                    seen_img_ids.add(image_id)

                    img_ids.append(image_id)
                    img_bytes_list.append(raw_bytes)
                    img_sha256_list.append(hashlib.sha256(raw_bytes).hexdigest())

                    img_info: dict[str, object] = {"text_ids": [article_id]}
                    image_url = img_meta.get("image_url")
                    if image_url:
                        img_info["url"] = image_url
                    width = img_meta.get("width")
                    height = img_meta.get("height")
                    if width:
                        img_info["width"] = width
                    if height:
                        img_info["height"] = height
                    caption_title = img_meta.get("caption_title", "")
                    if caption_title:
                        img_info["caption_title"] = caption_title
                    caption = img_meta.get("caption_text", "")
                    if caption:
                        img_info["caption"] = caption
                    md5 = img_meta.get("image_md5", "")
                    if md5:
                        img_info["md5"] = md5

                    imgdata_ids.append(image_id)
                    imgdata_infos.append(json.dumps(img_info, ensure_ascii=False))
                    imgdata_datas.append(str(caption))
                    imgdata_tags.append([])

                # Keep original remote image URLs for now.
                # html_data = rewrite_html_images(
                #     html_data,
                #     img_url_to_id,
                #     available_ids,
                # )
                _ = available_ids

                info: dict[str, object] = {
                    "format": "html",
                    "url": url,
                    "title": title,
                }
                if image_ids_for_article:
                    info["image_ids"] = image_ids_for_article
                original_url = entry.get("url", "")
                if original_url and original_url != url:
                    info["original_url"] = original_url
                for key in ("crawl_time", "crawl_type", "page_type", "part", "image_status"):
                    val = entry.get(key)
                    if val is not None and val != "":
                        info[key] = val

                text_ids.append(article_id)
                text_infos.append(json.dumps(info, ensure_ascii=False))
                text_datas.append(html_data)
                text_tags.append(tags)

                done = text_written + processed_in_part
                if log_interval > 0 and (processed_in_part % log_interval == 0 or processed_in_part == part_total):
                    elapsed = time.perf_counter() - t0
                    speed = done / elapsed if elapsed > 0 else 0.0
                    eta = (total_articles - done) / speed if speed > 0 else 0.0
                    log.info(
                        "  progress %d/%d (%.1f%%) | elapsed %s | avg %.2f art/s | eta %s",
                        done, total_articles, 100.0 * done / total_articles,
                        _fmt_seconds(elapsed), speed, _fmt_seconds(max(eta, 0.0)),
                    )

        mode = "overwrite" if part_idx == 1 else "append"
        lance.write_dataset(
            pa.table(
                {
                    "id": pa.array(text_ids, type=pa.string()),
                    "info": pa.array(text_infos, type=pa.string()),
                    "data": pa.array(text_datas, type=pa.large_string()),
                    "tags": pa.array(text_tags, type=pa.list_(pa.string())),
                },
                schema=TEXT_SCHEMA,
            ),
            str(text_lance),
            mode=mode,
        )

        if img_ids:
            lance.write_dataset(
                pa.table(
                    {
                        "id": pa.array(img_ids, type=pa.string()),
                        "image_bytes": pa.array(img_bytes_list, type=pa.binary()),
                        "sha256": pa.array(img_sha256_list, type=pa.string()),
                    },
                    schema=IMAGES_SCHEMA,
                ),
                str(images_lance),
                mode=mode,
                data_storage_version="2.0",
            )
            lance.write_dataset(
                pa.table(
                    {
                        "id": pa.array(imgdata_ids, type=pa.string()),
                        "info": pa.array(imgdata_infos, type=pa.string()),
                        "data": pa.array(imgdata_datas, type=pa.string()),
                        "tags": pa.array(imgdata_tags, type=pa.list_(pa.string())),
                    },
                    schema=IMAGE_LABELS_SCHEMA,
                ),
                str(image_labels_lance),
                mode=mode,
                data_storage_version="2.1",
            )

        text_written += len(text_ids)
        images_written += len(img_ids)
        log.info(
            "part done: %s text=%d new_images=%d | part %s | total %s",
            jf.name, len(text_ids), len(img_ids),
            _fmt_seconds(time.perf_counter() - part_t0),
            _fmt_seconds(time.perf_counter() - t0),
        )

    log.info(
        "Ingest finished: articles=%d images=%d | total %s",
        text_written, images_written, _fmt_seconds(time.perf_counter() - t0),
    )

    compact_t0 = time.perf_counter()
    compact_lance(text_lance, "text")
    if images_lance.is_dir():
        compact_lance(images_lance, "images")
    if image_labels_lance.is_dir():
        compact_lance(image_labels_lance, "image_labels")
    log.info(
        "Compact done in %s | output: %s",
        _fmt_seconds(time.perf_counter() - compact_t0),
        dst_dir,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Convert wiki part*.jsonl + part*.tar to Lance datasets",
    )
    parser.add_argument("src_dir", type=Path, help="Directory containing part*.jsonl and part*.tar")
    parser.add_argument("dst_dir", type=Path, help="Output dataset directory")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help=f"Progress log interval in articles (default: {DEFAULT_LOG_INTERVAL})",
    )
    args = parser.parse_args()

    run_streaming(args.src_dir, args.dst_dir, log_interval=max(1, args.log_interval))


if __name__ == "__main__":
    main()
