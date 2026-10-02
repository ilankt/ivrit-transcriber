"""Replace text files only after their complete contents have reached disk."""
from contextlib import contextmanager
import os
import tempfile


@contextmanager
def atomic_text_writer(path):
    directory = os.path.dirname(os.path.abspath(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, prefix=".ivrit-", delete=False,
        ) as stream:
            temporary = stream.name
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
