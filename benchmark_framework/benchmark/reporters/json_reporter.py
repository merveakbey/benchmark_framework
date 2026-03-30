import json
from pathlib import Path

from benchmark.reporters.base_reporter import BaseReporter


class JSONReporter(BaseReporter):
    def __init__(self, output_dir: str, filename: str = "report.json"):
        self.output_dir = Path(output_dir)
        self.filename = filename

    def write(self, report: dict) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / self.filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return str(output_path)