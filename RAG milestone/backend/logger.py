import logging

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Prevent adding multiple handlers if this is called many times
        handler = logging.StreamHandler()
        class ColorFormatter(logging.Formatter):
            COLOR_CODES = {
                'DEBUG': '\033[94m',
                'INFO': '\033[92m',
                'WARNING': '\033[93m',
                'ERROR': '\033[91m',
                'CRITICAL': '\033[95m',
            }
            RESET = '\033[0m'

            def format(self, record):
                color = self.COLOR_CODES.get(record.levelname, self.RESET)
                record.name = f'\033[96m{record.name}{self.RESET}'
                record.msg = f'{color}{record.msg}{self.RESET}'
                return super().format(record)

        formatter = ColorFormatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger