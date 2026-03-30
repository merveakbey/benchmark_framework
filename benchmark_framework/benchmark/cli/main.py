import argparse
import json
import sys

from benchmark.core.benchmark_runner import BenchmarkRunner
from benchmark.core.config_loader import ConfigLoader
from benchmark.utils.logging_utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="Benchmark Framework CLI")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    logger = get_logger()

    try:
        config = ConfigLoader.load(args.config)
        runner = BenchmarkRunner(config=config, logger=logger)
        runner.setup()
        report = runner.run()
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()