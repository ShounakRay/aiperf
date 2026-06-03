# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar, NoReturn

from aiperf.common.enums import (
    EnergyMetricUnit,
    GenericMetricUnit,
    MetricConsoleGroup,
    MetricFlags,
    PowerMetricUnit,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics import BaseDerivedMetric
from aiperf.metrics.metric_dicts import MetricResultsDict


class _ExternallyInjectedDerivedMetric(BaseDerivedMetric[float]):
    """Base for GPU power-efficiency totals that are externally injected.

    These metrics are not derived from `MetricResultsDict`; they are computed
    once per profiling phase by `GPUTelemetryAccumulator.compute_efficiency_metrics`
    and injected directly. `_derive_value` is intentionally non-functional;
    `MetricResultsProcessor.update_derived_metrics` is expected to catch
    `NoMetricValue` and skip the tag during its derivation walk. If this
    exception surfaces, that handler is missing.
    """

    __is_abstract__: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs) -> None:
        cls.__is_abstract__ = False
        return super().__init_subclass__(**kwargs)

    def _derive_value(self, metric_results: MetricResultsDict) -> NoReturn:
        raise NoMetricValue(
            f"Cannot derive '{self.tag}' from MetricResultsDict: this metric is "
            "externally injected by "
            "GPUTelemetryAccumulator.compute_efficiency_metrics. If this exception "
            "surfaces, the derivation walk is missing its NoMetricValue handler "
            "(see MetricResultsProcessor.update_derived_metrics)."
        )


# --- NVIDIA -----------------------------------------------------------------


class NvidiaTotalGpuPowerMetric(_ExternallyInjectedDerivedMetric):
    """Sum of average NVIDIA GPU power across all NVIDIA GPUs, in watts."""

    tag = "nvidia_total_gpu_power"
    header = "Total GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 900
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaTotalGpuEnergyMetric(_ExternallyInjectedDerivedMetric):
    """Sum of NVIDIA GPU energy consumed across all NVIDIA GPUs, in joules."""

    tag = "nvidia_total_gpu_energy"
    header = "Total GPU Energy"
    unit = EnergyMetricUnit.JOULE
    display_order = 901
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaOutputTokensPerJouleMetric(_ExternallyInjectedDerivedMetric):
    """Total output tokens divided by total NVIDIA GPU energy, in tokens per joule."""

    tag = "nvidia_output_tokens_per_joule"
    header = "Output Tokens per Joule"
    unit = GenericMetricUnit.TOKENS_PER_JOULE
    display_order = 902
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


class NvidiaEnergyPerUserMetric(_ExternallyInjectedDerivedMetric):
    """Total NVIDIA GPU energy divided by configured concurrency, in joules per user."""

    tag = "nvidia_energy_per_user"
    header = "Energy per User"
    unit = GenericMetricUnit.JOULES_PER_USER
    display_order = 903
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_NVIDIA


# --- AMD ---------------------------------------------------------------------


class AmdTotalGpuPowerMetric(_ExternallyInjectedDerivedMetric):
    """Sum of average AMD GPU power across all AMD GPUs, in watts."""

    tag = "amd_total_gpu_power"
    header = "Total GPU Power"
    unit = PowerMetricUnit.WATT
    display_order = 910
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdTotalGpuEnergyMetric(_ExternallyInjectedDerivedMetric):
    """Sum of AMD GPU energy consumed across all AMD GPUs, in joules."""

    tag = "amd_total_gpu_energy"
    header = "Total GPU Energy"
    unit = EnergyMetricUnit.JOULE
    display_order = 911
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdOutputTokensPerJouleMetric(_ExternallyInjectedDerivedMetric):
    """Total output tokens divided by total AMD GPU energy, in tokens per joule."""

    tag = "amd_output_tokens_per_joule"
    header = "Output Tokens per Joule"
    unit = GenericMetricUnit.TOKENS_PER_JOULE
    display_order = 912
    flags = MetricFlags.LARGER_IS_BETTER | MetricFlags.PRODUCES_TOKENS_ONLY
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD


class AmdEnergyPerUserMetric(_ExternallyInjectedDerivedMetric):
    """Total AMD GPU energy divided by configured concurrency, in joules per user."""

    tag = "amd_energy_per_user"
    header = "Energy per User"
    unit = GenericMetricUnit.JOULES_PER_USER
    display_order = 913
    flags = MetricFlags.NONE
    console_group = MetricConsoleGroup.GPU_POWER_EFFICIENCY_AMD
