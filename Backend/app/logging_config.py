import logging
from logging.handlers import RotatingFileHandler


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                "app.log",
                maxBytes=5_000_000,   # ~5MB per file
                backupCount=3,        # keep 3 old files, delete oldest after that
            ),
        ],
    )
