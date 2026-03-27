"""Convert wiki HTML rows to markdown with local image path rewrite."""

from __future__ import annotations

import json
import logging
import re
import signal
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin

import lance
from dcd_cli.pipe import PipeContext
from lxml.html import document_fromstring

from dataclawdev.tool.html import (
    PageMeta,
    make_cleaner,
    make_html_converter,
    make_md_converter,
)

log = logging.getLogger(__name__)

LOCAL_MEDIA_PREFIXES = ("images/", "media/")

THUMB_SIZE_RE = re.compile(r"/thumb/(.*)/\d+px-[^/]+$")
IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
IMG_SRC_RE = re.compile(r'(\bsrc=["\'])([^"\']+)(["\'])', re.IGNORECASE)
SRCSET_RE = re.compile(r'\s*\bsrcset=["\'][^"\']*["\']', re.IGNORECASE)
WIKIMEDIA_RE = re.compile(r"upload\.wikimedia\.org", re.IGNORECASE)
MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)([^)]*)\)")

_IMG_URL_MAP_CACHE: dict[str, dict[str, str]] = {}
_THREAD_TIMEOUT_WARNED = False


@dataclass
class ExtractResult:
    markdown: str
    simple_html: str
    meta: dict[str, str]


class ItemTimeoutError(RuntimeError):
    """Raised when one item exceeds conversion timeout."""


def normalize_url(url: str) -> str:
    """Strip protocol and normalize Wikimedia thumbnail URLs."""
    url = unquote(re.sub(r"^https?:", "", url).lstrip("/"))
    m = THUMB_SIZE_RE.search(url)
    if m:
        url = url[: m.start()] + "/" + m.group(1)
    return url


def run_extract_pipeline(
    source_html: str,
    url: str = "",
    *,
    remove_ref: bool = False,
) -> ExtractResult:
    """Run the same extraction pipeline used by ADP demo_parse_html."""
    tree = document_fromstring(source_html)
    meta = PageMeta(tree, url=url, remove_ref=remove_ref)

    cleaner = make_cleaner(meta)
    meta, content = cleaner.clean(tree)

    html_output = make_html_converter(meta).convert(deepcopy(content))
    md_output = make_md_converter(meta).convert(content)

    return ExtractResult(
        markdown=md_output,
        simple_html=html_output,
        meta=meta.to_dict(),
    )


def restore_local_paths(text: str, url: str) -> str:
    """Undo URL resolution for local dataset media paths."""
    if not url:
        return text
    for prefix in LOCAL_MEDIA_PREFIXES:
        resolved = urljoin(url, prefix)
        if resolved != prefix:
            text = text.replace(resolved, prefix)
    return text


def _resolve_dataset_dir(ctx: PipeContext) -> Path | None:
    if ctx.dataset_dir is not None:
        return Path(ctx.dataset_dir)
    config = ctx.config or {}
    cfg_path = config.get("dataset_dir", "")
    if not cfg_path:
        return None
    return Path(str(cfg_path))


def _load_img_url_map(dataset_dir: Path) -> dict[str, str]:
    """Load URL -> image_id mapping from image_labels.lance."""
    cache_key = str(dataset_dir.resolve())
    cached = _IMG_URL_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    labels_path = dataset_dir / "image_labels.lance"
    if not labels_path.is_dir():
        _IMG_URL_MAP_CACHE[cache_key] = {}
        return {}

    ds = lance.dataset(str(labels_path))
    tbl = ds.to_table(columns=["id", "info"])
    ids = tbl.column("id").to_pylist()
    infos = tbl.column("info").to_pylist()

    url_to_id: dict[str, str] = {}
    for image_id, info_raw in zip(ids, infos, strict=True):
        if not image_id or not info_raw:
            continue
        try:
            info = json.loads(info_raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(info, dict):
            continue
        url = info.get("url", "")
        if isinstance(url, str) and url:
            url_to_id[normalize_url(url)] = str(image_id)

    log.info("Loaded %d image URL mappings from %s", len(url_to_id), labels_path)
    _IMG_URL_MAP_CACHE[cache_key] = url_to_id
    return url_to_id


def rewrite_html_images(
    html: str,
    img_url_to_id: dict[str, str],
) -> tuple[str, list[str]]:
    """Rewrite Wikimedia image URLs to ``images/{id}`` when possible."""
    matched_ids: list[str] = []
    seen: set[str] = set()

    def replacer(m: re.Match[str]) -> str:
        tag = m.group(0)
        src_m = IMG_SRC_RE.search(tag)
        if not src_m:
            return tag
        src = src_m.group(2)
        if not WIKIMEDIA_RE.search(src):
            return tag

        image_id = img_url_to_id.get(normalize_url(src))
        if not image_id:
            return tag

        if image_id not in seen:
            seen.add(image_id)
            matched_ids.append(image_id)
        tag = SRCSET_RE.sub("", tag)
        return IMG_SRC_RE.sub(rf"\g<1>images/{image_id}\g<3>", tag)

    return IMG_TAG_RE.sub(replacer, html), matched_ids


def rewrite_markdown_images(
    markdown: str,
    img_url_to_id: dict[str, str],
) -> tuple[str, list[str]]:
    """Rewrite markdown image URLs to ``images/{id}`` when possible."""
    matched_ids: list[str] = []
    seen: set[str] = set()

    def replacer(m: re.Match[str]) -> str:
        alt_text = m.group(1)
        url = m.group(2)
        suffix = m.group(3)

        if url.startswith("images/"):
            image_id = url.removeprefix("images/")
            if image_id and image_id not in seen:
                seen.add(image_id)
                matched_ids.append(image_id)
            return m.group(0)

        if not WIKIMEDIA_RE.search(url):
            return m.group(0)

        image_id = img_url_to_id.get(normalize_url(url))
        if not image_id:
            return m.group(0)

        if image_id not in seen:
            seen.add(image_id)
            matched_ids.append(image_id)
        return f"![{alt_text}](images/{image_id}{suffix})"

    return MD_IMAGE_RE.sub(replacer, markdown), matched_ids


def fallback_markdown(source_html: str) -> str:
    """Best-effort plain-text markdown fallback for timeout/error cases."""
    try:
        tree = document_fromstring(source_html)
        text = tree.text_content()
    except Exception:  # noqa: BLE001
        return ""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n\n".join(lines)


def convert_markdown_with_timeout(
    source_html: str,
    *,
    url: str,
    remove_ref: bool,
    max_item_seconds: int,
) -> tuple[str, bool]:
    """Convert HTML to markdown with timeout and fallback."""
    global _THREAD_TIMEOUT_WARNED

    use_signal_timeout = (
        max_item_seconds > 0
        and threading.current_thread() is threading.main_thread()
    )

    if max_item_seconds > 0 and not use_signal_timeout and not _THREAD_TIMEOUT_WARNED:
        log.warning(
            "max_item_seconds=%s requested, but running in non-main thread; "
            "signal timeout disabled for this worker",
            max_item_seconds,
        )
        _THREAD_TIMEOUT_WARNED = True

    if not use_signal_timeout:
        try:
            result = run_extract_pipeline(
                source_html,
                url,
                remove_ref=remove_ref,
            )
            return restore_local_paths(result.markdown, url), False
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Convert error (%s), fallback markdown for url=%s",
                exc,
                url,
            )
            return fallback_markdown(source_html), True

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise ItemTimeoutError()

    old_handler: Any | None = None
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, max_item_seconds)
        result = run_extract_pipeline(
            source_html,
            url,
            remove_ref=remove_ref,
        )
        return restore_local_paths(result.markdown, url), False
    except ItemTimeoutError:
        log.warning(
            "Timeout after %ss, fallback markdown for url=%s",
            max_item_seconds,
            url,
        )
        return fallback_markdown(source_html), True
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Convert error (%s), fallback markdown for url=%s",
            exc,
            url,
        )
        return fallback_markdown(source_html), True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)


def map(batch: dict[str, list[Any]], ctx: PipeContext) -> dict[str, list[Any]]:
    """Convert source HTML rows in ``data`` to markdown."""
    config = ctx.config or {}
    remove_ref: bool = bool(config.get("remove_ref", False))
    max_item_seconds = int(config.get("max_item_seconds", 8))
    rewrite_images: bool = bool(config.get("rewrite_images", True))

    img_url_to_id: dict[str, str] = {}
    if rewrite_images:
        dataset_dir = _resolve_dataset_dir(ctx)
        if dataset_dir is not None:
            img_url_to_id = _load_img_url_map(dataset_dir)
        else:
            log.warning(
                "rewrite_images=true but dataset_dir is unavailable; "
                "skip image URL rewrite",
            )

    data_out: list[str] = []
    info_out: list[str] = []
    for source_html, info_raw in zip(
        batch["data"], batch["info"], strict=True,
    ):
        source_html = source_html or ""
        info_raw = info_raw or "{}"
        info: dict[str, Any] = (
            json.loads(info_raw)
            if isinstance(info_raw, str)
            else info_raw
        )
        if not isinstance(info, dict):
            info = {}

        if not source_html:
            data_out.append(source_html)
            info["format"] = "md"
            info_out.append(json.dumps(info, ensure_ascii=False))
            continue

        html_matched_ids: list[str] = []
        if img_url_to_id:
            html_for_convert, html_matched_ids = rewrite_html_images(
                source_html,
                img_url_to_id,
            )
        else:
            html_for_convert = source_html

        url = str(info.get("url", ""))
        markdown, fallback_used = convert_markdown_with_timeout(
            html_for_convert,
            url=url,
            remove_ref=remove_ref,
            max_item_seconds=max_item_seconds,
        )

        md_matched_ids: list[str] = []
        if img_url_to_id:
            markdown, md_matched_ids = rewrite_markdown_images(
                markdown,
                img_url_to_id,
            )
        data_out.append(markdown)

        info["format"] = "md"
        if fallback_used:
            info["md_fallback"] = True
        if img_url_to_id:
            info["image_path_scheme"] = "images/{id}"
            merged_ids: list[str] = []
            seen: set[str] = set()
            for image_id in html_matched_ids + md_matched_ids:
                if image_id and image_id not in seen:
                    seen.add(image_id)
                    merged_ids.append(image_id)
            if merged_ids:
                info["image_ids"] = merged_ids
        info_out.append(json.dumps(info, ensure_ascii=False))

    return {**batch, "data": data_out, "info": info_out}
