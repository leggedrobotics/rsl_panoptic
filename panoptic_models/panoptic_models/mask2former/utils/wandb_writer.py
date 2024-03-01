# import package
from collections import defaultdict
import wandb
from detectron2.utils.events import EventWriter, get_event_storage


class WandbWriter(EventWriter):
    """
    Write metrices to weights and biases

    "data_time": 0.008433341979980469,
    "iteration": 19,
    "loss": 1.9228371381759644,
    "loss_box_reg": 0.050025828182697296,
    "loss_classifier": 0.5316952466964722,
    "loss_mask": 0.7236229181289673,
    "loss_rpn_box": 0.0856662318110466,
    "loss_rpn_cls": 0.48198649287223816,
    "lr": 0.007173333333333333,
    "time": 0.25401854515075684
    """

    def __init__(self, window_size=20):
        """
        Args:
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._window_size = window_size
        self._last_write = -1

    def write(self):
        storage = get_event_storage()
        to_save = defaultdict(dict)

        for k, (v, iter) in storage.latest_with_smoothing_hint(
            self._window_size
        ).items():
            # keep scalars that have not been written
            if iter <= self._last_write:
                continue
            to_save[iter][k] = v
        if len(to_save):
            all_iters = sorted(to_save.keys())
            self._last_write = max(all_iters)

        for itr, scalars_per_iter in to_save.items():
            scalars_per_iter["iteration"] = itr
            wandb.log(scalars_per_iter)
