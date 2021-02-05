from typing import Any
from typing import Dict
from typing import List

import optuna
from optuna._imports import try_import

with try_import() as _imports:
    from pytorch_pfn_extras.config import Config
    from pytorch_pfn_extras.config_types import optuna_types
    import yaml


def suggest_ppe_optuna_types(trial: optuna.trial.Trial, ppe_config_str: str) -> str:

    _imports.check()

    config = yaml.load(ppe_config_str)
    keys = _retrieve_optuna_keys(config)
    ppe_config = Config(config, optuna_types(trial))
    # Suggest all params.
    for k in keys:
        ppe_config[k]
    return yaml.dump(_traverse_and_replace(trial.params, config))


def _traverse_and_replace(params: Dict[str, Any], item) -> Any:
    if isinstance(item, dict):
        return_item = {}
        for name, value in item.items():
            if isinstance(value, dict):
                if "type" in value and value["type"].startswith("optuna_"):
                    # Replace optuna_types
                    return_item[name] = params[value["name"]]
                else:
                    return_item[name] = _traverse_and_replace(params, value)
            else:
                return_item[name] = _traverse_and_replace(params, value)
        return return_item
    elif isinstance(item, list):
        return_item = []
        for value in item:
            if isinstance(value, dict):
                if "type" in value and value["type"].startswith("optuna_"):
                    # Replace optuna_types
                    return_item.append(params[value["name"]])
                else:
                    return_item.append(_traverse_and_replace(params, value))
            else:
                return_item.append(_traverse_and_replace(params, value))
        return return_item
    else:
        return item


def _retrieve_optuna_keys(item: Any, prefix: str = "") -> List[str]:
    ret_keys: List[str] = []
    if isinstance(item, dict):
        for name, value in item.items():
            if isinstance(value, dict):
                if "type" in value and value["type"].startswith("optuna_"):
                    # Replace optuna_types
                    ret_keys.append(f"{prefix}/{name}")
                else:
                    ret_keys += _retrieve_optuna_keys(value, f"{prefix}/{name}")
            else:
                ret_keys += _retrieve_optuna_keys(value, f"{prefix}/{name}")
    elif isinstance(item, list):
        for i, value in enumerate(item):
            if isinstance(value, dict):
                if "type" in value and value["type"].startswith("optuna_"):
                    # Replace optuna_types
                    ret_keys.append(f"{prefix}/{i}")
                else:
                    ret_keys += _retrieve_optuna_keys(value, f"{prefix}/{i}")
            else:
                ret_keys += _retrieve_optuna_keys(value, f"{prefix}/{i}")
    return ret_keys
