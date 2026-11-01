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
"""Checkpointable postponed-question ordering, without an unbounded retry loop."""

import numpy as np
from torch.utils.data import Dataset, Sampler


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        result = dict(self.dataset[index])
        result["iop_dataset_index"] = index
        return result


class DeferredSampler(Sampler):
    """Visit every question once per epoch, moving deferred questions to the next epoch's tail."""

    def __init__(self, dataset, shuffle=True, seed=42):
        self.size, self.shuffle, self.seed = len(dataset), shuffle, seed
        self.epoch, self.position, self.order = 0, 0, []
        self.deferred = set()

    def __len__(self):
        return self.size

    def __iter__(self):
        if not self.order or self.position == len(self.order):
            order = list(range(self.size))
            if self.shuffle:
                np.random.default_rng(self.seed + self.epoch).shuffle(order)
            self.order = [i for i in order if i not in self.deferred] + [i for i in order if i in self.deferred]
            self.position = 0
            self.epoch += 1
        while self.position < len(self.order):
            index = self.order[self.position]
            self.position += 1
            yield index

    def defer(self, index, failed=True):
        index = int(index)
        if not 0 <= index < self.size:
            raise ValueError("IOP dataset index outside the sampler")
        if failed:
            self.deferred.add(index)
        else:
            self.deferred.discard(index)

    def state_dict(self):
        return {
            "size": self.size,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order.copy(),
            "deferred": sorted(self.deferred),
        }

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("IOP checkpoint dataset size changed")
        if sorted(state["order"]) != list(range(self.size)) and state["order"]:
            raise ValueError("Invalid IOP sampler order")
        if not 0 <= state["position"] <= len(state["order"]):
            raise ValueError("Invalid IOP sampler position")
        self.epoch, self.position = state["epoch"], state["position"]
        self.order, self.deferred = list(state["order"]), set(state["deferred"])
