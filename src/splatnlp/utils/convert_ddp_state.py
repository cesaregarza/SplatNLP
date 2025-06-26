from collections import OrderedDict


def convert_ddp_state(state_dict: dict) -> dict:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module.") :]
        new_state_dict[k] = v
    return new_state_dict
