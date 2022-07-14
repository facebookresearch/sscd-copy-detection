# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Dict, List, Any, Union, Tuple


class SamplerType(str, Enum):
    FIXED = ("fixed",)
    CHOICE = ("choice",)
    TUPLE = "tuple"
    UNIFORM = "uniform"
    UNIFORMINT = "uniformint"


class Sampler(Callable):
    @classmethod
    def from_config(cls, sampler_spec_or_value: Any) -> "Sampler":
        if not isinstance(sampler_spec_or_value, Mapping) or (
            "sampler_type" not in sampler_spec_or_value
        ):
            return FixedValueSampler(sampler_spec_or_value)
        sampler_spec = sampler_spec_or_value
        sampler_type = SamplerType(sampler_spec["sampler_type"])
        if sampler_type == SamplerType.FIXED:
            return FixedValueSampler.from_config(sampler_spec)
        elif sampler_type == SamplerType.CHOICE:
            return ChoiceSampler.from_config(sampler_spec)
        elif sampler_type == SamplerType.TUPLE:
            return TupleSampler.from_config(sampler_spec)
        elif sampler_type == SamplerType.UNIFORM:
            return UniformSampler.from_config(sampler_spec)
        elif sampler_type == SamplerType.UNIFORMINT:
            return UniformIntSampler.from_config(sampler_spec)
        else:
            raise ValueError(f"Unknown sampler type {sampler_type}")


class FixedValueSampler(Sampler):
    """
    Trivial sampler that returns a fixed value
    """

    def __init__(self, value: Any):
        self._value = value

    def __call__(self):
        return self._value

    @classmethod
    def from_config(cls, sampler_spec: Dict[str, Any]) -> "FixedValueSampler":
        assert (
            "value" in sampler_spec
        ), f"Fixed sampler value not specified: {sampler_spec}"
        value = sampler_spec["value"]
        return cls(value=value)


class ChoiceSampler(Sampler):
    """
    Produces samples from the given population of values uniformly or
    based on provided weights. Config for uniform sampling should
    look like:
    {
        "sampler_type": "choice",
        "values": ["a", "b", "c"]
    }
    Config for categorical distribution sampling should look like:
    {
        "sampler_type": "choice",
        "values": { "a": 0.5, "b": 0.2, "c": 0.3 }
    }
    """

    def __init__(self, values: Union[Dict, List]):
        if isinstance(values, list):
            self._values = values
            self._weights = None
        elif isinstance(values, dict):
            ks, vs = zip(*values.items())
            self._values = list(ks)
            self._weights = list(vs)
        else:
            raise ValueError(
                f"Choice sampler values are of type {type(values)},"
                f" expected either list or dict"
            )

    def __call__(self):
        sample = random.choices(self._values, self._weights)[0]
        return sample

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ChoiceSampler":
        assert (
            "values" in config
        ), f"Choice sampler values not specified in config: {config}"
        values = config["values"]
        return cls(values=values)


class UniformSampler(Sampler):
    """
    Samples values from a uniform distribution given by lower and upper bounds
    """

    def __init__(self, low: float, high: float):
        self._low = low
        self._high = high

    def __call__(self):
        sample = random.uniform(self._low, self._high)
        return sample

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "UniformSampler":
        assert (
            "low" in config
        ), f'Uniform sampler lower bound ("low") not specified in config: {config}'
        assert (
            "high" in config
        ), f'Uniform sampler upper bound ("high") not specified in config: {config}'
        return cls(low=config["low"], high=config["high"])


class UniformIntSampler(Sampler):
    """
    Samples values from a uniform int distribution given by lower and upper bounds
    """

    def __init__(self, low: int, high: int):
        self._low = low
        self._high = high

    def __call__(self):
        sample = random.randint(self._low, self._high)
        return sample

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "UniformIntSampler":
        assert "low" in config, (
            f'Uniform int sampler lower bound ("low") not specified in '
            f"config: {config}"
        )
        assert "high" in config, (
            f'Uniform int sampler upper bound ("high") not specified in '
            f"config: {config}"
        )
        return cls(low=config["low"], high=config["high"])


class TupleSampler(Sampler):
    """
    Sample a tuple of values given a collection of samplers
    """

    def __init__(self, samplers: Tuple):
        self._samplers = samplers

    def __call__(self):
        samples = tuple(sampler() for sampler in self._samplers)
        return samples

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TupleSampler":
        assert (
            "samplers" in config
        ), f'Tuple samplers ("samplers") not specified in config: {config}'
        samplers = [
            Sampler.from_config(sampler_spec) for sampler_spec in config["samplers"]
        ]
        return cls(samplers=samplers)
