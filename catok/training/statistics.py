"""Action normalization utilities used by training and evaluation."""

import json

import numpy as np


class ActionNormalizerBase:
    def __init__(self, stats_path):
        with open(stats_path, "r") as file:
            self.stats = json.load(file)

    def normalize(self, action):
        raise NotImplementedError

    def denormalize(self, action):
        raise NotImplementedError


class ZScoreNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path, eps=1e-8, action_dim=7, clip=True):
        super().__init__(stats_path)
        self.action_dim = action_dim
        self.mean = np.asarray(self.stats["action"]["mean"])
        self.std = np.asarray(self.stats["action"]["std"])
        self.eps = eps

    def normalize(self, action):
        head = (action[..., : self.action_dim] - self.mean[: self.action_dim]) / (
            self.std[: self.action_dim] + self.eps
        )
        return np.concatenate([head, action[..., self.action_dim :]], axis=-1)

    def denormalize(self, action):
        head = action[..., : self.action_dim] * (self.std[: self.action_dim] + self.eps)
        head += self.mean[: self.action_dim]
        return np.concatenate([head, action[..., self.action_dim :]], axis=-1)


class PercentileSymmetricNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path, clip=True):
        super().__init__(stats_path)
        q01 = np.asarray(self.stats["action"]["q01"])
        q99 = np.asarray(self.stats["action"]["q99"])
        self.scale = np.maximum(np.abs(q01), np.abs(q99))
        self.clip = clip

    def normalize(self, action):
        output = action / self.scale
        return np.clip(output, -1.0, 1.0) if self.clip else output

    def denormalize(self, action):
        return action * self.scale


class AbsMaxNormalizer(PercentileSymmetricNormalizer):
    def __init__(self, stats_path, clip=True):
        ActionNormalizerBase.__init__(self, stats_path)
        self.scale = np.asarray(self.stats["action"]["abs_max"])
        self.clip = clip


class MinMaxNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path):
        super().__init__(stats_path)
        self.minimum = np.asarray(self.stats["action"]["min"])
        self.maximum = np.asarray(self.stats["action"]["max"])

    def normalize(self, action):
        return 2 * (action - self.minimum) / (self.maximum - self.minimum) - 1

    def denormalize(self, action):
        return (action + 1) * (self.maximum - self.minimum) / 2 + self.minimum


class QQNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path, clip=True, transform="identity", stats_key="action", action_dim=None):
        super().__init__(stats_path)
        self.q01 = np.asarray(self.stats[stats_key]["q01"])
        self.q99 = np.asarray(self.stats[stats_key]["q99"])
        self.clip = clip
        self.action_dim = action_dim or len(self.q01)

    def normalize(self, action):
        output = np.array(action, copy=True)
        dimensions = self.action_dim
        output[..., :dimensions] = (
            (output[..., :dimensions] - self.q01[:dimensions])
            / (self.q99[:dimensions] - self.q01[:dimensions] + 1e-7)
            * 2.0
            - 1.0
        )
        return np.clip(output, -1.0, 1.0) if self.clip else output

    def denormalize(self, action):
        output = np.array(action, copy=True)
        dimensions = self.action_dim
        output[..., :dimensions] = (
            (output[..., :dimensions] + 1.0)
            / 2.0
            * (self.q99[:dimensions] - self.q01[:dimensions] + 1e-7)
            + self.q01[:dimensions]
        )
        return output


class ScaleOnlyNormalizer(ActionNormalizerBase):
    def __init__(self, stats_path, method="q99", clip=True, scale_factor=1.0, no_gripper=False):
        super().__init__(stats_path)
        self.scale_factor = scale_factor
        if method == "q99":
            q01 = np.asarray(self.stats["action"]["q01"])
            q99 = np.asarray(self.stats["action"]["q99"])
            self.scale = np.maximum(np.abs(q01), np.abs(q99))
        elif method == "abs_max":
            self.scale = np.asarray(self.stats["action"]["abs_max"])
        else:
            raise ValueError(f"Unknown scale method: {method}")
        if no_gripper:
            self.scale = self.scale[:-1]
        self.clip = clip

    def normalize(self, action):
        output = action / self.scale
        if self.clip:
            output = np.clip(output, -1.0, 1.0)
        return output * self.scale_factor

    def denormalize(self, action):
        return action / self.scale_factor * self.scale


def build_action_normalizer(method, normalizer_config):
    """Construct a configured action normalizer."""
    if method is None or normalizer_config is None:
        return None
    normalizers = {
        "zscore": ZScoreNormalizer,
        "percentile": PercentileSymmetricNormalizer,
        "absmax": AbsMaxNormalizer,
        "minmax": MinMaxNormalizer,
        "scale_only": ScaleOnlyNormalizer,
        "qq": QQNormalizer,
    }
    try:
        normalizer = normalizers[method]
    except KeyError as error:
        raise ValueError(f"Unknown normalizer method: {method}") from error
    return normalizer(**normalizer_config)
