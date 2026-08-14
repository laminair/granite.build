#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Bash environment's cleanup_nohup.

Regression coverage for the bug where cancelling a build running on the Bash
environment never terminated the OS process tree launch_nohup started: Bash
had no cleanup_<launcher_type> hook, so TargetStepRun._cleanup found nothing
to call and the launched process (and its descendants, e.g. a server
listening on a port) kept running indefinitely after cancellation.
"""

import asyncio
import os

import pytest


def _make_bash():
    from gbserver.environment.bash import Bash

    return Bash(event_q=asyncio.Queue())


async def _spawn_group(script: str):
    """Start script in its own session/process group, mirroring launch_nohup."""
    return await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        script,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_cleanup_nohup_kills_process_group():
    """cleanup_nohup terminates the whole tree, not just the top-level process.

    Mirrors the real shape: a parent shell that backgrounds a child ("mlx_lm.server")
    and then sleeps, same as llmb_bash_jobsub.sh -> llmb_bash_wrapper.sh -> the
    workload. Both must die.
    """
    bash = _make_bash()
    launch_id = "launch-cleanup-1"

    process = await _spawn_group("sleep 60 & sleep 60 & wait")
    bash._launched_processes[launch_id] = process
    pgid = process.pid

    assert _pgid_alive(pgid), "process group should be alive right after spawn"

    await asyncio.wait_for(bash.cleanup_nohup(launch_id=launch_id), timeout=15)

    await asyncio.sleep(0.2)
    assert not _pgid_alive(pgid), "process group should be dead after cleanup_nohup"
    assert launch_id not in bash._launched_processes


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_cleanup_nohup_falls_back_to_sigkill(monkeypatch):
    """A process that ignores SIGTERM is still reaped, via the SIGKILL fallback."""
    import gbserver.environment.bash as bash_module

    # Keep the test fast: don't wait out the real grace period for the SIGTERM path.
    monkeypatch.setattr(bash_module, "SIGTERM_GRACE_PERIOD_SECONDS", 0.1)

    bash = _make_bash()
    launch_id = "launch-cleanup-2"

    process = await _spawn_group("trap '' TERM; sleep 60")
    bash._launched_processes[launch_id] = process
    pgid = process.pid

    assert _pgid_alive(pgid)

    await asyncio.wait_for(bash.cleanup_nohup(launch_id=launch_id), timeout=15)

    await asyncio.sleep(0.2)
    assert not _pgid_alive(pgid), "SIGKILL fallback should have reaped the group"


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_cleanup_nohup_no_process_is_noop():
    bash = _make_bash()
    await bash.cleanup_nohup(launch_id="nonexistent-launch")


@pytest.mark.standalone
@pytest.mark.asyncio
async def test_cleanup_nohup_handles_already_exited_process():
    bash = _make_bash()
    launch_id = "launch-cleanup-3"

    process = await _spawn_group("true")
    bash._launched_processes[launch_id] = process
    await process.wait()

    await asyncio.wait_for(bash.cleanup_nohup(launch_id=launch_id), timeout=15)
    assert launch_id not in bash._launched_processes
