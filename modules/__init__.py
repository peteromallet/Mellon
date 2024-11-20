from os import scandir
from importlib import import_module
import logging
logger = logging.getLogger('mellon')

logger.debug("Loading modules...")

MODULE_MAP = {}

for m in scandir("modules"):
    if m.is_dir() and not m.name.startswith(("__", ".")) and m.name not in globals():
        MODULE_MAP[m.name] = import_module(f"modules.{m.name}").MODULE_MAP
        logger.info(f"Loaded module: {m.name}")
