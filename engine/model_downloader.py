"""
On-demand model downloader for Ivrit Transcriber.
Downloads English Whisper models from HuggingFace Hub when not present locally.
"""
import os


def download_ct2_model(
    repo_id: str,
    dest_dir: str,
    progress_cb=None,
    cancel_check=None,
) -> None:
    """
    Download all files for a CTranslate2 model from HuggingFace, one file at a time.

    Args:
        repo_id: HuggingFace repository ID
        dest_dir: Local directory to populate
        progress_cb: Optional callable(int 0-100) for progress updates
        cancel_check: Optional callable() -> bool; raises InterruptedError if True
    """
    from huggingface_hub import list_repo_files, hf_hub_download

    os.makedirs(dest_dir, exist_ok=True)

    files = list(list_repo_files(repo_id, repo_type="model"))
    total = max(len(files), 1)

    for i, filename in enumerate(files):
        if cancel_check and cancel_check():
            raise InterruptedError("Download canceled")
        if progress_cb:
            progress_cb(int(i * 100 / total))
        _hf_download(repo_id, filename, dest_dir)

    if progress_cb:
        progress_cb(100)


def download_ggml_file(
    repo_id: str,
    filename: str,
    dest_dir: str,
    progress_cb=None,
    cancel_check=None,
) -> None:
    """
    Download a single GGML model file from HuggingFace.

    Args:
        repo_id: HuggingFace repository ID
        filename: Filename within the repository
        dest_dir: Local directory to save the file into
        progress_cb: Optional callable(int 0-100) for progress updates
        cancel_check: Optional callable() -> bool; raises InterruptedError if True
    """
    if cancel_check and cancel_check():
        raise InterruptedError("Download canceled")

    os.makedirs(dest_dir, exist_ok=True)

    if progress_cb:
        progress_cb(0)

    _hf_download(repo_id, filename, dest_dir)

    if progress_cb:
        progress_cb(100)


def _hf_download(repo_id: str, filename: str, local_dir: str) -> None:
    """Download one file from HuggingFace, disabling symlinks for Windows compatibility."""
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
    except TypeError:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
        )
