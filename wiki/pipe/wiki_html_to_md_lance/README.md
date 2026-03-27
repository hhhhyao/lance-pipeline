# wiki_html_to_md_lance

Convert wiki `text.lance` HTML content into markdown with local image-path
rewriting.

## What it does

- Reads `text.lance` rows (`id`, `data`, `info`, `tags`).
- Uses ADP HTML cleaning + markdown conversion (`dataclawdev.tool.html`).
- Rewrites Wikimedia image URLs to local `images/{id}` in two stages:
  - HTML stage: rewrite `<img src=...>` before markdown conversion.
  - Markdown stage: rewrite remaining `![...](...)` URLs after conversion.
- URL matching follows wiki normalization (`normalize_url`) to handle
  thumbnail/full-size URL variants.
- Sets `info.format = "md"` in output rows.
- Writes output `text.lance` only. Local run reuses `images.lance` and
  `image_labels.lance` via symlink from source dataset.

## Local run (multiprocess)

```bash
python3 wiki/pipe/wiki_html_to_md_lance/run_local.py \
  workspace/html_lance/wiki_0320_en_has_pic \
  workspace/md_lance/wiki_0320_en_has_pic \
  --workers 48 --batch-size 64
```

Or use wrapper script:

```bash
bash wiki/pipe/run_local.sh \
  workspace/html_lance/wiki_0320_en_has_pic \
  workspace/md_lance/wiki_0320_en_has_pic
```

## Remote update

```bash
bash wiki/pipe/upload_pipe.sh update wiki/pipe/wiki_html_to_md_lance \
  "update wiki html->md conversion and image rewrite"
```
