import argparse
import json
from pathlib import Path

from benchmark.reporters.comparison_csv_reporter import ComparisonCSVReporter
from benchmark.reporters.comparison_markdown_reporter import ComparisonMarkdownReporter
from benchmark.reporters.accuracy_loss_reporter import AccuracyLossReporter


def load_report(report_path: str) -> dict:
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_row(report: dict) -> dict:
    run_metadata = report.get("run_metadata", {})
    evaluation_summary = report.get("evaluation_summary", {})
    latency_summary = report.get("latency_summary", {})
    monitoring_summary = report.get("monitoring_summary", {})
    monitor_0 = monitoring_summary.get("monitor_0", {})

    return {
        "run_name": run_metadata.get("run_name"),
        "backend": run_metadata.get("backend"),
        "precision_mode": run_metadata.get("precision"),
        "map_50": evaluation_summary.get("map_50"),
        "map_50_95": evaluation_summary.get("map_50_95"),
        "precision_metric": evaluation_summary.get("precision"),
        "recall": evaluation_summary.get("recall"),
        "tp": evaluation_summary.get("tp"),
        "fp": evaluation_summary.get("fp"),
        "fn": evaluation_summary.get("fn"),
        "num_predictions": evaluation_summary.get("num_predictions"),
        "num_ground_truths": evaluation_summary.get("num_ground_truths"),
        "avg_inference_ms": latency_summary.get("stage_breakdown_ms", {}).get("inference", {}).get("mean"),
        "avg_full_pipeline_ms": latency_summary.get("stage_breakdown_ms", {}).get("full_pipeline", {}).get("mean"),
        "inference_fps": latency_summary.get("inference_fps"),
        "avg_fps": latency_summary.get("avg_fps"),
        "avg_preprocess_ms": latency_summary.get("stage_breakdown_ms", {}).get("preprocess", {}).get("mean"),
        "avg_postprocess_ms": latency_summary.get("stage_breakdown_ms", {}).get("postprocess", {}).get("mean"),
        "avg_cpu_percent": monitor_0.get("cpu_percent", {}).get("mean"),
        "peak_cpu_percent": monitor_0.get("cpu_percent", {}).get("peak"),
        "avg_process_ram_mb": monitor_0.get("process_ram_mb", {}).get("mean"),
        "peak_process_ram_mb": monitor_0.get("process_ram_mb", {}).get("peak"),
        "avg_system_ram_used_mb": monitor_0.get("system_ram_used_mb", {}).get("mean"),
        "peak_system_ram_used_mb": monitor_0.get("system_ram_used_mb", {}).get("peak"),
        "avg_temperature_c": monitor_0.get("temperature_c", {}).get("mean"),
        "peak_temperature_c": monitor_0.get("temperature_c", {}).get("peak"),
        "output_dir": run_metadata.get("output_dir"),
    }


def collect_run_dirs_from_reports(report_paths: list[str]) -> list[str]:
    run_dirs = []
    for report_path in report_paths:
        report_file = Path(report_path).resolve()
        run_dir = report_file.parent
        if run_dir.is_dir():
            run_dirs.append(str(run_dir))
    return run_dirs


def append_accuracy_loss_section_to_markdown(md_path: Path, markdown_section: str) -> None:
    if not md_path.exists():
        return

    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(markdown_section)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple benchmark runs")
    parser.add_argument("--reports", nargs="+", required=True, help="Paths to report.json files")
    parser.add_argument("--output-dir", required=True, help="Directory to write comparison outputs")
    args = parser.parse_args()

    rows = []
    for report_path in args.reports:
        report = load_report(report_path)
        rows.append(extract_row(report))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "comparison_summary.csv"
    md_path = output_dir / "comparison_report.md"

    csv_reporter = ComparisonCSVReporter(str(csv_path))
    md_reporter = ComparisonMarkdownReporter(str(md_path))

    csv_reporter.write(rows)
    md_reporter.write(rows)

    print(f"[INFO] Comparison CSV written: {csv_path}")
    print(f"[INFO] Comparison Markdown written: {md_path}")

    # ===============================
    # ACCURACY LOSS ANALYSIS
    # ===============================
    run_dirs = collect_run_dirs_from_reports(args.reports)

    accuracy_reporter = AccuracyLossReporter(run_dirs=run_dirs)
    acc_result = accuracy_reporter.run(
        output_json_path=str(output_dir / "accuracy_loss_report.json"),
        output_csv_path=str(output_dir / "accuracy_loss_summary.csv"),
    )

    append_accuracy_loss_section_to_markdown(md_path, acc_result["markdown_section"])

    print(f"[INFO] Accuracy loss JSON written: {output_dir / 'accuracy_loss_report.json'}")
    print(f"[INFO] Accuracy loss CSV written: {output_dir / 'accuracy_loss_summary.csv'}")
    print(f"[INFO] Accuracy loss pair count: {acc_result['pair_count']}")


if __name__ == "__main__":
    main()