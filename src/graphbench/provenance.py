"""Fingerprint source dependencies, independently of checkout line endings.

Artifact bytes remain hashed exactly. Source fingerprints deliberately normalize
CRLF so moving the checkout between Windows and Unix does not change identity.
Historical results keep their original fingerprints and are never relabelled.
"""

import hashlib
from pathlib import Path

GENERATION_SOURCES = ("data.py", "generate.py", "history.py", "paths.py", "provenance.py")
EVALUATION_SOURCES = (
    "artifacts.py",
    "cache.py",
    "constants.py",
    "evaluate.py",
    "metrics.py",
    "models.py",
    "provenance.py",
    "summaries.py",
)


def source_digest(directory: Path, names) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update((directory / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def evaluation_code_hash() -> str:
    return source_digest(Path(__file__).parent, EVALUATION_SOURCES)
