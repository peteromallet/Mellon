import logging
import os
from config import config
import torch # warm up since we are going to use it anyway

if config.hf['cache_dir']:
    os.environ['HF_HOME'] = config.hf['cache_dir']

# initialize logging
logging.basicConfig(level=config.log['level'], format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y%m%d %H.%M.%S")
logger = logging.getLogger('mellon')

# load modules
from modules import MODULE_MAP

from server.web import WebServer
web_server = WebServer(MODULE_MAP, **config.server)

logger.info(f"""\x1b[33;20m
╔══════════════════════╗
║  Welcome to Mellon!  ║
╚══════════════════════╝\x1b[0m
Speak Friend and Enter: http://{config.server['host']}:{config.server['port']}""")
web_server.run()
