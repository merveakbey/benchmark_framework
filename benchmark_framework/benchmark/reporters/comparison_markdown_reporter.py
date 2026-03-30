from pathlib import Path


class ComparisonMarkdownReporter:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)

    def write(self, rows: list[dict]) -> str:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("# Benchmark Comparison Report\n")

        if not rows:
            lines.append("Karşılaştırılacak veri bulunamadı.\n")
        else:
            lines.append("## Backend Özeti\n")
            lines.append(
                "| Backend | Precision | mAP@0.5 | mAP@0.5:0.95 | Precision Metric | Recall | "
                "Avg Inference (ms) | Inference FPS | Avg Full Pipeline (ms) | Avg FPS | "
                "Avg CPU % | Avg Process RAM (MB) | Avg Temp (°C) |"
            )
            lines.append(
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            )

            for row in rows:
                lines.append(
                    f"| {row.get('backend')} "
                    f"| {row.get('precision_mode')} "
                    f"| {row.get('map_50')} "
                    f"| {row.get('map_50_95')} "
                    f"| {row.get('precision_metric')} "
                    f"| {row.get('recall')} "
                    f"| {row.get('avg_inference_ms')} "
                    f"| {row.get('inference_fps')} "
                    f"| {row.get('avg_full_pipeline_ms')} "
                    f"| {row.get('avg_fps')} "
                    f"| {row.get('avg_cpu_percent')} "
                    f"| {row.get('avg_process_ram_mb')} "
                    f"| {row.get('avg_temperature_c')} |"
                )

            lines.append("\n## Kısa Yorum\n")

            best_map50 = max(rows, key=lambda x: float(x.get("map_50", 0) or 0))
            best_speed = min(rows, key=lambda x: float(x.get("avg_inference_ms", 1e9) or 1e9))
            best_avg_fps = max(rows, key=lambda x: float(x.get("avg_fps", 0) or 0))

            lines.append(
                f"- En yüksek **mAP@0.5**: `{best_map50.get('backend')}` "
                f"({best_map50.get('map_50')})"
            )
            lines.append(
                f"- En düşük **inference süresi**: `{best_speed.get('backend')}` "
                f"({best_speed.get('avg_inference_ms')} ms)"
            )
            lines.append(
                f"- En yüksek **uçtan uca FPS**: `{best_avg_fps.get('backend')}` "
                f"({best_avg_fps.get('avg_fps')})"
            )

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return str(self.output_path)