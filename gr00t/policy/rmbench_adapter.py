# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMBench-specific policy protocol adapters."""

from typing import Any

import numpy as np

from .policy import BasePolicy, PolicyWrapper


def _validate_current_actual_arm(current_actual_arm: np.ndarray) -> None:
    if not isinstance(current_actual_arm, np.ndarray):
        raise TypeError(
            f"current actual arm state must be an ndarray, got {type(current_actual_arm)}"
        )
    if current_actual_arm.ndim != 3 or current_actual_arm.shape[1] != 1:
        raise ValueError(
            "current actual arm state must have shape [B,1,D], "
            f"got {current_actual_arm.shape}"
        )
    if not np.isfinite(current_actual_arm).all():
        raise FloatingPointError("current actual arm state contains NaN or Inf")


def relative_arm_to_absolute(
    relative_arm: np.ndarray, current_actual_arm: np.ndarray
) -> np.ndarray:
    """Convert a relative arm horizon using one non-cumulative actual-qpos base."""
    if not isinstance(relative_arm, np.ndarray):
        raise TypeError(f"relative arm action must be an ndarray, got {type(relative_arm)}")
    _validate_current_actual_arm(current_actual_arm)
    if relative_arm.ndim != 3:
        raise ValueError(f"relative arm action must have shape [B,H,D], got {relative_arm.shape}")
    if (
        relative_arm.shape[0] != current_actual_arm.shape[0]
        or relative_arm.shape[2] != current_actual_arm.shape[2]
    ):
        raise ValueError(
            "relative arm action and current actual arm state must match in batch and "
            f"dimension, got {relative_arm.shape} and {current_actual_arm.shape}"
        )
    if not np.isfinite(relative_arm).all():
        raise FloatingPointError("relative arm action contains NaN or Inf")

    absolute_arm = relative_arm + current_actual_arm
    if not np.isfinite(absolute_arm).all():
        raise FloatingPointError("absolute arm action contains NaN or Inf after conversion")
    return absolute_arm


class RMBenchAbsoluteArmPolicyWrapper(PolicyWrapper):
    """End GR00T relative-arm semantics at the PPU protocol boundary."""

    _ARM_STATE_KEY = "joint_position"
    _ARM_ACTION_KEY = "joint_position"
    _ARM_DIM = 12

    def __init__(self, policy: BasePolicy, *, strict: bool = True):
        super().__init__(policy, strict=strict)

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            current_actual_arm = observation["state"][self._ARM_STATE_KEY]
        except KeyError as exc:
            raise KeyError(
                f"observation is missing state.{self._ARM_STATE_KEY} actual qpos"
            ) from exc
        _validate_current_actual_arm(current_actual_arm)
        if current_actual_arm.shape[2] != self._ARM_DIM:
            raise ValueError(
                f"RMBench actual arm state must be 12D, got {current_actual_arm.shape}"
            )

        raw_action, diagnostics = self.policy.get_action(observation, options)
        if self._ARM_ACTION_KEY not in raw_action:
            raise KeyError(f"model action is missing {self._ARM_ACTION_KEY!r}")

        relative_arm = raw_action[self._ARM_ACTION_KEY]
        absolute_arm = relative_arm_to_absolute(relative_arm, current_actual_arm)
        external_action = dict(raw_action)
        external_action[self._ARM_ACTION_KEY] = absolute_arm

        diagnostics = dict(diagnostics or {})
        diagnostics["arm_boundary_adapter"] = {
            "state_source": f"observation.state.{self._ARM_STATE_KEY}",
            "raw_model_semantics": "relative",
            "external_semantics": "absolute",
            "conversion_formula": "absolute[t+k]=actual[t]+relative[t+k]",
            "cumulative_conversion": False,
            "current_actual_arm": current_actual_arm[:, 0, :].tolist(),
            "raw_relative_arm_first": relative_arm[:, 0, :].tolist(),
            "returned_absolute_arm_first": absolute_arm[:, 0, :].tolist(),
        }
        return external_action, diagnostics

    def check_observation(self, observation: dict[str, Any]) -> None:
        self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        self.policy.check_action(action)

    def get_modality_config(self):
        return getattr(self.policy, "get_modality_config", lambda: {})()
