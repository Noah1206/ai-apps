"""Export Pydantic contracts as JSON Schema."""

from pathlib import Path

from busan_lab.cli import export_schemas

if __name__ == "__main__":
    export_schemas(Path("reports/schemas"))
