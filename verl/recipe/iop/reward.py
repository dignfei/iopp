# Copyright 2026 Individual Contributor: d fei
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
"""Binary outcome adapter; formatting credit is not correctness."""

from verl.utils.reward_score import default_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    result = default_compute_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
    if isinstance(result, dict):
        result = dict(result)
        correctness = result["acc"] if "acc" in result else result["score"]
    else:
        correctness = result
        result = {}
    result["score"] = result["acc"] = float(correctness == 1)
    return result
