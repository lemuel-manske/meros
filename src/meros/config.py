"""Execution options; diagnostics never determine stage completeness."""

from dataclasses import dataclass


@dataclass
class ExecutionOptions:
    diagnostics: bool = False
