# evaluator.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np


class Evaluator:
    """
    Unified evaluation for both async_fl and sync_fl settings.

    Usage:
        overall, per_client = Evaluator().evaluate_online(clients, online_mask)

    - overall: Dict[str, float]
        * macro means for all reported metrics (loss/accuracy/precision/recall/f1)
        * optional micro means for accuracy/loss (suffix: _micro), weighted by clients' val_dataset_len
        * optional accuracy quantiles (accuracy_p10, accuracy_p50, accuracy_p90)
    - per_client: Dict[client_id, Dict[str, float]]
        metrics returned by client.evaluate_model()
    """

    def __init__(
        self,
        include_micro: bool = True,
        include_quantiles: bool = True,
        quantiles: Tuple[float, float, float] = (0.1, 0.5, 0.9),
    ):
        self.include_micro = include_micro
        self.include_quantiles = include_quantiles
        self.quantiles = quantiles

    # ------------- public -------------
    def evaluate_online(
        self,
        clients: List,
        online_mask: List[bool],
    ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        online_ids = [i for i, on in enumerate(online_mask) if on]
        if not online_ids:
            return {}, {}

        per_client: Dict[int, Dict[str, float]] = {}
        val_sizes: Dict[int, int] = {}

        # collect metrics per online client
        for cid in online_ids:
            m = clients[cid].evaluate_model()
            per_client[cid] = m
            # requires your Client to expose val_dataset_len (present in your base class)
            val_sizes[cid] = int(getattr(clients[cid], "val_dataset_len", 0))

        overall = self._macro_average(per_client)

        if self.include_micro:
            overall.update(self._micro_avg_acc_loss(per_client, val_sizes))

        if self.include_quantiles:
            self._attach_accuracy_quantiles(overall, per_client)

        return overall, per_client

    def evaluate_all(self, clients: List) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
        """Convenience for sync_fl baseline: evaluate all clients as online."""
        mask = [True] * len(clients)
        return self.evaluate_online(clients, mask)

    # ------------- helpers -------------
    @staticmethod
    def _macro_average(per_client: Dict[int, Dict[str, float]]) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        n = len(per_client)
        if n == 0:
            return {}
        for m in per_client.values():
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + float(v)
        return {k: v / n for k, v in agg.items()}

    @staticmethod
    def _micro_avg_acc_loss(
        per_client: Dict[int, Dict[str, float]], sample_sizes: Dict[int, int]
    ) -> Dict[str, float]:
        # weighted by validation set sizes
        keys = ("accuracy", "loss")
        out: Dict[str, float] = {}
        weights = {cid: sample_sizes.get(cid, 0) for cid in per_client.keys()}
        total = sum(weights.values())
        if total <= 0:
            return out
        for key in keys:
            num = 0.0
            for cid, m in per_client.items():
                w = weights[cid]
                num += float(m.get(key, 0.0)) * w
            out[f"{key}_micro"] = num / total
        return out

    def _attach_accuracy_quantiles(
        self,
        overall: Dict[str, float],
        per_client: Dict[int, Dict[str, float]],
    ) -> None:
        accs = [float(m.get("accuracy", 0.0)) for m in per_client.values()]
        if not accs:
            return
        qs = np.quantile(np.array(accs, dtype=float), self.quantiles, method="linear")
        # attach readable keys
        for qv, qname in zip(qs, self._qnames()):
            overall[qname] = float(qv)

    def _qnames(self) -> List[str]:
        names = []
        for q in self.quantiles:
            p = int(round(100 * q))
            names.append(f"accuracy_p{p}")
        return names