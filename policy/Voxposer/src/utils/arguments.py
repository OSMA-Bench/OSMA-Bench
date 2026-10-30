"""load YAML config file"""
import os
import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def _deep_merge(base, override):
    """Recursively merge override into base dict (override wins)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_config(config_path=None, task_type=None):
    if config_path is None:
        config_path = 'src/configs/robocasa_config.yaml'
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    config = load_config(config_path)

    # merge task-type specific overrides into base config
    if task_type is not None and task_type in config:
        overrides = config.pop(task_type)
        config = _deep_merge(config, overrides)

    # remove other task-type sections
    for key in ['navigation', 'manipulation']:
        config.pop(key, None)

    class ConfigDict(dict):
        def __init__(self, config):
            """recursively build config"""
            self.config = config
            for key, value in config.items():
                if isinstance(value, str) and value.lower() == 'none':
                    value = None
                if isinstance(value, dict):
                    self[key] = ConfigDict(value)
                else:
                    self[key] = value
        def __getattr__(self, key):
            return self[key]
        def __setattr__(self, key, value):
            self[key] = value
        def __delattr__(self, key):
            del self[key]
        def __getstate__(self):
            return self.config
        def __setstate__(self, state):
            self.config = state
            self.__init__(state)
    config = ConfigDict(config)
    return config
