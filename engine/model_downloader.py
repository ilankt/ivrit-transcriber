"""Revision-pinned model downloads with checksum verification and repair."""
import hashlib
import json
import os
from pathlib import Path

from core.storage import atomic_text_writer

CT2_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.json")


def _check_cancel(cancel_check):
    if cancel_check and cancel_check():
        raise InterruptedError("Download canceled")


def _manifest(repo_id, filenames):
    from huggingface_hub import model_info
    info = model_info(repo_id, files_metadata=True)
    siblings = {item.rfilename: item for item in info.siblings}
    files = []
    for name in filenames:
        item = siblings.get(name)
        if item is None or not item.size or not (item.lfs or item.blob_id):
            raise ValueError(f"Missing verification metadata for {repo_id}/{name}")
        files.append(dict(name=name, size=item.size,
                          algorithm="sha256" if item.lfs else "git-sha1",
                          digest=item.lfs.sha256 if item.lfs else item.blob_id))
    if not info.sha:
        raise ValueError("Model revision could not be resolved")
    return dict(repo_id=repo_id, revision=info.sha, files=files)


def _matches(path, item, cancel_check):
    _check_cancel(cancel_check)
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    digest = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
    if item["algorithm"] == "git-sha1":
        digest.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            _check_cancel(cancel_check)
            digest.update(block)
    return digest.hexdigest() == item["digest"]


def _fetch_verified(manifest, item, target, cancel_check):
    if _matches(target, item, cancel_check):
        return
    staging = target.parent / ".ivrit_downloads" / manifest["repo_id"].replace("/", "--") / manifest["revision"]
    staging.mkdir(parents=True, exist_ok=True)
    staged = staging / item["name"]
    if not _matches(staged, item, cancel_check):
        # Re-download corrupt staging data even if Hub cache metadata says it is current.
        _hf_download(manifest["repo_id"], item["name"], str(staging), manifest["revision"], staged.exists())
        _check_cancel(cancel_check)
        if not _matches(staged, item, cancel_check):
            raise ValueError(f"Checksum verification failed for {item['name']}; try Verify / Repair again.")
    _check_cancel(cancel_check)
    os.replace(staged, target)


def download_ct2_model(repo_id, dest_dir, progress_cb=None, cancel_check=None):
    _check_cancel(cancel_check)
    manifest = _manifest(repo_id, CT2_FILES)
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    incomplete = destination / ".ivrit-incomplete"
    with atomic_text_writer(incomplete) as stream:
        stream.write(manifest["revision"])
    for index, item in enumerate(manifest["files"]):
        if progress_cb:
            progress_cb(index * 100 // len(manifest["files"]))
        target = destination / item["name"]
        _fetch_verified(manifest, item, target, cancel_check)
        item["mtime_ns"] = target.stat().st_mtime_ns
    _check_cancel(cancel_check)
    with atomic_text_writer(destination / ".ivrit-model.json") as stream:
        json.dump(manifest, stream, indent=2)
    incomplete.unlink()
    if progress_cb:
        progress_cb(100)


def download_ggml_file(repo_id, filename, dest_dir, progress_cb=None, cancel_check=None, target_name=None):
    _check_cancel(cancel_check)
    manifest = _manifest(repo_id, [filename])
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / (target_name or filename)
    if progress_cb:
        progress_cb(0)
    item = manifest["files"][0]
    _fetch_verified(manifest, item, target, cancel_check)
    _check_cancel(cancel_check)
    item["name"], item["mtime_ns"] = target.name, target.stat().st_mtime_ns
    with atomic_text_writer(str(target) + ".verified.json") as stream:
        json.dump(manifest, stream, indent=2)
    if progress_cb:
        progress_cb(100)


def _hf_download(repo_id, filename, local_dir, revision, force_download=False):
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir,
                    revision=revision, force_download=force_download)
