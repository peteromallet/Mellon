from os import scandir
from importlib import import_module
import logging
logger = logging.getLogger('mellon')

logger.debug("Loading modules...")

MODULE_MAP = {}

for m in scandir("modules"):
    if m.is_dir() and not m.name.startswith(("__", ".")) and m.name not in globals():
        MODULE_MAP[m.name] = import_module(f"modules.{m.name}").MODULE_MAP
        logger.debug(f"Loaded module: {m.name}")

logger.debug("Loading custom modules...")

for m in scandir("custom"):
    if m.is_dir() and not m.name.startswith(("__", ".")) and m.name not in globals():
        MODULE_MAP[f"{m.name}.custom"] = import_module(f"custom.{m.name}").MODULE_MAP
        logger.debug(f"Loaded custom module: {m.name}")
