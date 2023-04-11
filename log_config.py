import logging
import logging.config
from logging.config import dictConfig

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "[%(asctime)s - %(name)s - %(levelname)s - %(funcName)s: %(lineno)d] - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {"console_handler": {"class": "logging.StreamHandler", "formatter": "default",},},
    "root": {"level": "INFO"},
    "loggers": {"example": {"handlers": ["console_handler"], "propagate": True,}},
}


def configure_logger(log_file=False, console=True, log_level="DEBUG"):

    logger = logging.getLogger("example")

    dictConfig(LOGGING_DEFAULT_CONFIG)

    if not log_file:
        pass
    else:
        logging.basicConfig(
            filename="logs/logs.text",
            filemode="a",
            format="[%(asctime)s - %(name)s - %(levelname)s - %(funcName)s: %(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=getattr(logging, log_level),
        )

    if not console:
        pass
    else:
        logger.addHandler(logging.StreamHandler())

    if not log_level:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(getattr(logging, log_level))

    return logger
