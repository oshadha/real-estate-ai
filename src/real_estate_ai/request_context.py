import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

_request_id: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)

_request_id_pattern = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def get_request_id() -> str | None:
    return _request_id.get()


def resolve_request_id(header_value: str | None) -> str:
    if header_value is not None and _request_id_pattern.fullmatch(header_value):
        return header_value

    return str(uuid4())


@contextmanager
def bind_request_id(request_id: str) -> Iterator[None]:
    token = _request_id.set(request_id)

    try:
        yield
    finally:
        _request_id.reset(token)
