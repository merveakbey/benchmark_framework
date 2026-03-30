from pathlib import Path


class MonitorCSVReporter:
    def __init__(self, output_dir: str, filename: str = "monitor_trace.csv"):
        self.output_dir = Path(output_dir)
        self.filename = filename

    def write(self, monitor) -> str | None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / self.filename

        if hasattr(monitor, "export_csv"):
            return monitor.export_csv(str(output_path))

        return None