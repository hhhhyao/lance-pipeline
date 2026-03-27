# wiki/pipe

Pipe definitions and helper scripts for wiki processing.

## Contents

- `wiki_html_to_md_lance/`: Pipe package (`manifest.yaml`, `__init__.py`,
  local runner, requirements).
- `run_local.sh`: Wrapper for local HTML-Lance -> MD-Lance conversion.
- `upload_pipe.sh`: Validate and register/update pipe on remote ADP.

## Notes

- Current pipeline writes only `text.lance` for output dataset.
- `images.lance` and `image_labels.lance` are reused from upstream dataset
  by symlink in local run.
