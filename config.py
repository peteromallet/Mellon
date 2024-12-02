import configparser
import logging
import os

class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.server = {
            'host': self.config.get('server', 'host', fallback='127.0.0.1'),
            'port': self.config.getint('server', 'port', fallback=8080),
            'cors': self.config.getboolean('server', 'cors', fallback=False),
            'cors_route': self.config.get('server', 'cors_route', fallback='*'),
        }

        self.log = {
            'level': getattr(logging, self.config.get('logging', 'level', fallback='INFO').upper()),
        }

        self.hf = {
            'token': self.config.get('huggingface', 'token', fallback=None),
            'cache_dir': self.config.get('huggingface', 'cache_dir', fallback=None),
        }

config = Config()
