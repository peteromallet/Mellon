import logging
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

log_level = config.get('logging', 'level', fallback='INFO')
log_level = getattr(logging, log_level.upper())

server_cfg = {
    'host': config.get('server', 'host', fallback='127.0.0.1'),
    'port': config.getint('server', 'port', fallback=8080),
    'cors': config.getboolean('server', 'cors', fallback=False),
    'cors_route': config.get('server', 'cors_route', fallback='*'),
}

# initialize logging
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger('mellon')

# load modules
from modules import MODULE_MAP

from server.web import WebServer
web_server = WebServer(MODULE_MAP, **server_cfg)
web_server.run()
