class BenchmarkError(Exception):
    """Base exception for benchmark framework."""


class ConfigError(BenchmarkError):
    """Raised when the benchmark configuration is invalid."""


class RegistryError(BenchmarkError):
    """Raised when a component cannot be created from registry."""
