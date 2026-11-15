"""Shared output protection for reporting scripts."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


def check_output(path, *, overwrite=False, protected_paths=()):
    """Reject existing outputs by default and always protect input files."""
    path = Path(path)

    for source in protected_paths:
        source = Path(source)
        if path.resolve() == source.resolve() or (
            path.exists() and source.exists() and path.samefile(source)
        ):
            raise ValueError(
                f"output must not replace an input file: {source}"
            )

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {path}; choose a different filename "
            "or pass --overwrite"
        )


@contextmanager
def open_output(
    path,
    *,
    overwrite=False,
    protected_paths=(),
    binary=False,
    encoding="utf-8-sig",
    newline="",
):
    """Replace an existing output only after successful writing."""
    path = Path(path)
    protected_paths = tuple(protected_paths)

    check_output(
        path,
        overwrite=overwrite,
        protected_paths=protected_paths,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    options = (
        {}
        if binary
        else {"encoding": encoding, "newline": newline}
    )

    if not overwrite:
        handle = path.open("xb" if binary else "x", **options)
        try:
            with handle:
                yield handle
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return

    handle = tempfile.NamedTemporaryFile(
        mode="wb" if binary else "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        **options,
    )
    temporary = Path(handle.name)

    try:
        with handle:
            yield handle

        check_output(
            path,
            overwrite=True,
            protected_paths=protected_paths,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)