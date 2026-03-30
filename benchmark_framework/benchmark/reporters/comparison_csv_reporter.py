import csv
from pathlib import Path


class ComparisonCSVReporter:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)

    def write(self, rows: list[dict]) -> str:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if not rows:
            with open(self.output_path, "w", encoding="utf-8", newline="") as f:
                f.write("")
            return str(self.output_path)

        fieldnames = list(rows[0].keys())

        with open(self.output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return str(self.output_path)