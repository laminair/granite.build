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

"""repetition-check end to end on IBM BlueVela SLURM.

SUCCESS proves that the GPU image greedy-decodes the eight prompts from an HF checkpoint and registers
the repetition report. The rung is the 350m student itself, so no export is needed.

Run it from the step directory::

    GB_STEP_BLUEVELA_BUILD=1 make -C steps/train/distill/repetition-check/skypilot test TESTS=test/slurm-bluevela

IBM infrastructure through the local space, so marked ``ibm`` and extended-only. It
also needs an explicit opt-in: ``make test`` runs every test in the step directory, and
without one a routine run would submit real jobs to a shared cluster. Skips unless
``~/.ssh/ibm-bluevela.key`` exists (the key the shared environment's
``BV_SSH_PRIVATE_KEY`` space secret is created from), and in SPS CI/CD, which has no
key on the host.
"""

import os
from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

pytestmark = [pytest.mark.skypilot_integration, pytest.mark.ibm]

# The key the BV_SSH_PRIVATE_KEY space secret is created from; the environment reads
# the secret (cluster_ssh_configs IdentityKey), not this path.
_BLUEVELA_KEY = Path("~/.ssh/ibm-bluevela.key").expanduser()
_OPT_IN = "GB_STEP_BLUEVELA_BUILD"


@extended_testing_only
# No xdist_group: skypilot_integration already puts it in the serialized
# "skypilot" group, and under --dist=loadgroup any other group is a UsageError.
@pytest.mark.skipif(
    os.environ.get(_OPT_IN, "").lower() not in ("1", "true", "yes"),
    reason=f"Set {_OPT_IN}=1 to run the real BlueVela build",
)
@pytest.mark.skipif(
    not _BLUEVELA_KEY.is_file(),
    reason=f"BlueVela granitebuild SSH key not found at {_BLUEVELA_KEY}",
)
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="No BlueVela SSH key on SPS CI/CD hosts",
)
class TestSkypilotSlurmBluevelaTrainDistillRepetitionCheck(AbstractYamlBuildRunnerTest):
    """repetition-check on BlueVela SLURM, via Pyxis."""

    def _get_yaml_spec_dir(self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml."""
        return get_test_data_dir_for(__file__)

    @pytest.mark.timeout(7500)  # above buildtest.yaml's timeout_minutes (120)
    def test_runner(self):
        """Run the happy-path build with room for the buildtest timeout."""
        super().test_runner()
