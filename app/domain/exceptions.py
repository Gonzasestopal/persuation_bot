class DomainError(Exception):
    """Base class for all domain-level errors.

    These represent business or validation failures that occur within
    the application's core logic, independent of transport concerns.
    Raise subclasses of this in services, repositories, or adapters
    when the problem is related to business rules or domain state.
    """
    pass


class ConfigError(Exception):
    """Raised when the system is misconfigured.

    Use this for missing or invalid environment variables, API keys,
    or other critical configuration values that prevent the app from
    starting or operating correctly.
    """
    pass
