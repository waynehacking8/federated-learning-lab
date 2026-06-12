"""MNIST partitioning utilities for federated experiments.

Four partition schemes that systematically vary Non-IID severity:

    iid(num_clients)
    label_skew(num_clients, classes_per_client)
    dirichlet(num_clients, alpha)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def iid(targets: Sequence[int], num_clients: int, seed: int = 0) -> list[list[int]]:
    """Equal-size random split of dataset indices across clients."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    perm = rng.permutation(n)
    # np.array_split distributes the remainder across the first few clients.
    return [list(map(int, chunk)) for chunk in np.array_split(perm, num_clients)]


def label_skew(
    targets: Sequence[int],
    num_clients: int,
    classes_per_client: int,
    seed: int = 0,
) -> list[list[int]]:
    """Each client receives samples from a subset of classes only.

    Steps:
      1. Group indices by class.
      2. Assign ``classes_per_client`` classes to each client uniformly at random.
      3. Split each class's index pool equally across the clients holding it.
    """
    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    classes = np.unique(targets_arr)
    n_classes = len(classes)

    # Step 1: index pool per class (shuffled for randomized assignment).
    class_indices: dict[int, np.ndarray] = {}
    for c in classes:
        pool = np.where(targets_arr == c)[0]
        rng.shuffle(pool)
        class_indices[int(c)] = pool

    # Step 2: pick `classes_per_client` classes for each client.
    client_classes: list[list[int]] = [
        list(rng.choice(classes, size=classes_per_client, replace=False))
        for _ in range(num_clients)
    ]

    # Ensure every class is held by at least one client.  Naively replacing
    # a random client's slot can itself orphan the displaced class (and
    # silently drop all of its samples in step 3), so only displace classes
    # that have >= 2 holders, and never give a client a duplicate class.
    if num_clients * classes_per_client < n_classes:
        raise ValueError(
            f"cannot cover {n_classes} classes with "
            f"{num_clients} clients x {classes_per_client} classes each"
        )
    holder_counts: dict[int, int] = {int(c): 0 for c in classes}
    for cls_list in client_classes:
        for c in cls_list:
            holder_counts[int(c)] += 1
    for c in classes:
        c = int(c)
        if holder_counts[c] > 0:
            continue
        candidates = [
            (client_id, slot)
            for client_id, cls_list in enumerate(client_classes)
            if c not in (int(x) for x in cls_list)
            for slot, held in enumerate(cls_list)
            if holder_counts[int(held)] >= 2
        ]
        if not candidates:
            raise ValueError(f"unable to assign a holder for class {c}")
        client_id, slot = candidates[int(rng.integers(len(candidates)))]
        holder_counts[int(client_classes[client_id][slot])] -= 1
        client_classes[client_id][slot] = c
        holder_counts[c] = 1

    # Step 3: for each class, distribute its indices across the clients that hold it.
    holders: dict[int, list[int]] = {int(c): [] for c in classes}
    for client_id, cls_list in enumerate(client_classes):
        for c in cls_list:
            holders[int(c)].append(client_id)

    parts: list[list[int]] = [[] for _ in range(num_clients)]
    for c, holder_ids in holders.items():
        if not holder_ids:
            continue
        chunks = np.array_split(class_indices[c], len(holder_ids))
        for client_id, chunk in zip(holder_ids, chunks):
            parts[client_id].extend(int(i) for i in chunk)

    # Reproducibility: shuffle each client's own pool.
    for p in parts:
        rng.shuffle(p)
    return parts


def dirichlet(
    targets: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int = 0,
) -> list[list[int]]:
    """Dirichlet-alpha partition (Hsu, Qi, Brown 2019).

    For each class c:
        p_c ~ Dir(alpha * ones(num_clients))
        Assign class c's samples to clients in proportion to p_c.
    """
    rng = np.random.default_rng(seed)
    targets_arr = np.asarray(targets)
    classes = np.unique(targets_arr)

    parts: list[list[int]] = [[] for _ in range(num_clients)]
    for c in classes:
        idx = np.where(targets_arr == c)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        # Integer split points along the class's index pool.
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        chunks = np.split(idx, cuts)
        for client_id, chunk in enumerate(chunks):
            parts[client_id].extend(int(i) for i in chunk)

    for p in parts:
        rng.shuffle(p)
    return parts
