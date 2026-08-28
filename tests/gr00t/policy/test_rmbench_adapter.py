# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMBench external arm-semantics boundary tests."""

import socket
import threading

import numpy as np
import pytest

from gr00t.eval.run_gr00t_server import (
    RMBENCH_ADAPTER_VERSION,
    ServerConfig,
    _build_server_metadata,
)
from gr00t.policy.rmbench_adapter import (
    RMBenchAbsoluteArmPolicyWrapper,
    relative_arm_to_absolute,
)
from gr00t.policy.server_client import MsgSerializer, PolicyClient, PolicyServer


def _current_actual_arm() -> np.ndarray:
    return np.arange(1.0, 13.0, dtype=np.float32)[None, None, :]


def _relative_horizon() -> np.ndarray:
    return np.stack(
        (
            np.full(12, 0.1, dtype=np.float32),
            np.full(12, 0.2, dtype=np.float32),
        ),
        axis=0,
    )[None, :, :]


class _RelativePolicy:
    strict = False

    def __init__(self):
        self.calls = 0
        self.gripper = np.array([[[0.25, 0.75], [0.5, 1.0]]], dtype=np.float32)

    def get_action(self, observation, options=None):
        self.calls += 1
        relative = _relative_horizon() * self.calls
        return {
            "joint_position": relative,
            "gripper_close": self.gripper,
        }, {"inference_count": self.calls}

    def reset(self, options=None):
        return {}

    def check_observation(self, observation):
        pass

    def check_action(self, action):
        pass

    def get_metadata(self):
        return {
            "adapter_version": "gr00t_policy_adapter_v3",
            "arm_action_semantics": "absolute",
            "gripper_action_semantics": "absolute",
        }


def _observation() -> dict:
    return {"state": {"joint_position": _current_actual_arm()}}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_relative_to_absolute_uses_same_base_for_every_horizon_step():
    current = _current_actual_arm()
    relative = _relative_horizon()

    absolute = relative_arm_to_absolute(relative, current)

    np.testing.assert_allclose(absolute[:, 0], current[:, 0] + 0.1)
    np.testing.assert_allclose(absolute[:, 1], current[:, 0] + 0.2)
    assert not np.allclose(absolute[:, 1], absolute[:, 0] + 0.2)


def test_relative_to_absolute_preserves_action_shape():
    absolute = relative_arm_to_absolute(_relative_horizon(), _current_actual_arm())
    assert absolute.shape == (1, 2, 12)


def test_boundary_adapter_leaves_gripper_untouched():
    raw_policy = _RelativePolicy()
    adapter = RMBenchAbsoluteArmPolicyWrapper(raw_policy, strict=False)

    action, diagnostics = adapter.get_action(_observation())

    assert action["gripper_close"] is raw_policy.gripper
    np.testing.assert_array_equal(action["gripper_close"], raw_policy.gripper)
    assert diagnostics["arm_boundary_adapter"]["cumulative_conversion"] is False


@pytest.mark.parametrize("invalid_source", ["state", "action"])
def test_relative_to_absolute_rejects_nonfinite_values(invalid_source):
    current = _current_actual_arm()
    relative = _relative_horizon()
    if invalid_source == "state":
        current[0, 0, 0] = np.nan
    else:
        relative[0, 0, 0] = np.inf

    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        relative_arm_to_absolute(relative, current)


def test_boundary_rejects_nonfinite_state_before_inference():
    raw_policy = _RelativePolicy()
    adapter = RMBenchAbsoluteArmPolicyWrapper(raw_policy, strict=False)
    observation = _observation()
    observation["state"]["joint_position"][0, 0, 0] = np.nan

    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        adapter.get_action(observation)

    assert raw_policy.calls == 0


def test_converted_response_is_idempotent_on_wire():
    raw_policy = _RelativePolicy()
    adapter = RMBenchAbsoluteArmPolicyWrapper(raw_policy, strict=False)
    port = _free_port()
    options = {
        "episode_id": "semantic-episode",
        "request_id": "semantic-request",
        "source_step_id": 4,
        "dry_run": True,
    }

    with PolicyServer(adapter, host="127.0.0.1", port=port) as server:
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        with PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000) as client:
            first = client.call_endpoint(
                "get_action", {"observation": _observation(), "options": options}
            )
            retry = client.call_endpoint(
                "get_action", {"observation": _observation(), "options": options}
            )
            client.kill_server()
        thread.join(timeout=2)

    assert raw_policy.calls == 1
    assert MsgSerializer.to_bytes(first) == MsgSerializer.to_bytes(retry)
    np.testing.assert_array_equal(first[0]["joint_position"], retry[0]["joint_position"])
    for field in ("episode_id", "request_id", "source_step_id"):
        assert first[1][field] == options[field]


def test_adapter_v3_identity_declares_external_state_and_action_semantics():
    config = ServerConfig(
        state_arm_semantics="actual_qpos",
        state_gripper_semantics="physical",
        arm_action_semantics="absolute",
        gripper_action_semantics="absolute",
        relative_arm_to_absolute_boundary=True,
    )

    metadata = _build_server_metadata(config)

    assert RMBENCH_ADAPTER_VERSION == "gr00t_policy_adapter_v3"
    assert metadata["adapter_version"] == "gr00t_policy_adapter_v3"
    assert metadata["state_arm_semantics"] == "actual_qpos"
    assert metadata["state_gripper_semantics"] == "physical"
    assert metadata["arm_action_semantics"] == "absolute"
    assert metadata["gripper_action_semantics"] == "absolute"
    assert metadata["relative_arm_to_absolute_boundary"] is True
