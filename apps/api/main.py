"""Uvicorn entrypoint for the installed AUDIRE application."""

from audire.api import create_app

app = create_app()

__all__ = ["app", "create_app"]
