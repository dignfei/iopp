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
"""Prepare externally verified repair records for verl's existing SFT trainer."""

import argparse
import json
from pathlib import Path

from recipe.iop.algorithms import repair_messages


def prepare_sft(input_path, output_path):
    """Convert question/failed/reference/repaired JSONL to messages Parquet."""
    import pandas as pd

    rows = []
    for line in Path(input_path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("verified") is not True:
            raise ValueError("IOP SFT requires verified=true for every human-provided repair")
        for key in ("failed", "reference", "repaired"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"IOP SFT requires nonempty {key}")
        question = record["question"]
        if isinstance(question, str):
            question = [{"role": "user", "content": question}]
        messages = repair_messages(question, record["failed"], record["reference"])
        messages.append({"role": "assistant", "content": record["repaired"]})
        rows.append({"messages": messages})
    if not rows:
        raise ValueError("IOP SFT input is empty")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(f"Prepared {prepare_sft(args.input, args.output)} verified repair records")


if __name__ == "__main__":
    main()
