from configparser import ConfigParser
import os 

def init_config(config_path=None) -> ConfigParser:
    config = ConfigParser()
    if not config_path:
        env = os.getenv("ENVIRONMENT")
        if not env:
            raise 'must set environment'
        config_path = 'configs/{}.ini'.format(env)

    config.read(config_path)
    return config