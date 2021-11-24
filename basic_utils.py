import os
import json
import zipfile
import numpy as np
import pickle
import random

def shuffle_list(*ls):
  l =list(zip(*ls))
  random.shuffle(l)
  return zip(*l)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def save_lines(list_of_str, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(list_of_str))


def read_lines(filepath):
    with open(filepath, "r") as f:
        return [e.strip("\n") for e in f.readlines()]


def mkdirp(p):
    if not os.path.exists(p):
        os.makedirs(p)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def load_pretrained_weight(model, state_dict, start_prefix=''):

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = ({} if metadata is None
                          else metadata.get(prefix[:-1], {}))
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys,
            unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=start_prefix)

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from "
                    "pretrained model: {}".format(
                        model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in "
                    "{}: {}".format(
                        model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for '
                            '{}:\n\t{}'.format(
                                model.__class__.__name__,
                                "\n\t".join(error_msgs)))
    return model