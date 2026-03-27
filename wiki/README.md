# wiki

Wiki data conversion and pipe workflows.

## Structure

- `wiki_jsonl_tar_to_lance.py`: Convert wiki tar+jsonl source into HTML Lance.
- `run_pipeline.sh`: Entrypoint for the raw-to-HTML conversion.
- `pipe/`: HTML-to-Markdown pipe implementation and helper scripts.

## Typical flow

1. Build HTML Lance dataset under `workspace/html_lance/...`.
2. Run `wiki/pipe/wiki_html_to_md_lance/run_local.py` (or `wiki/pipe/run_local.sh`)
   to build Markdown Lance under `workspace/md_lance/...`.
3. Upload/update the pipe with `wiki/pipe/upload_pipe.sh` and submit remote jobs.
