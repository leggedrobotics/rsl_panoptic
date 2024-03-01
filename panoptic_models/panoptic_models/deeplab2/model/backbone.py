import functools
from typing import Any, Dict, Text, Tuple

from absl import logging
import tensorflow as tf

from panoptic_seg.deeplab2 import config_pb2
from panoptic_seg.deeplab2.model import builder


class PanopticDeeplabBackbone(tf.keras.Model):
    def __init__(self, config: config_pb2.ExperimentOptions):
        """
        Initialize a panoptic deeplab model backbone for test-time adaptation.
        Args:
            config: A config_pb2.ExperimentOptions configuration.
            dataset_descriptor: A dataset.DatasetDescriptor.
        """
        super(PanopticDeeplabBackbone, self).__init__(name="panoptic_deeplab_backbone")
        if config.trainer_options.solver_options.use_sync_batchnorm:
            logging.info("Synchronized Batchnorm is used.")
            bn_layer = functools.partial(
                tf.keras.layers.experimental.SyncBatchNormalization,
                momentum=config.trainer_options.solver_options.batchnorm_momentum,
                epsilon=config.trainer_options.solver_options.batchnorm_epsilon,
            )
        else:
            logging.info("Standard (unsynchronized) Batchnorm is used.")
            bn_layer = functools.partial(
                tf.keras.layers.BatchNormalization,
                momentum=config.trainer_options.solver_options.batchnorm_momentum,
                epsilon=config.trainer_options.solver_options.batchnorm_epsilon,
            )

        self._encoder = builder.create_encoder(
            config.model_options.backbone,
            bn_layer,
            conv_kernel_weight_decay=(
                config.trainer_options.solver_options.weight_decay / 2
            ),
        )

    def call(self, input_tensor: tf.Tensor, training: bool = False) -> Dict[Text, Any]:
        """
        Performs a forward pass
        Args:
            input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
                          width, channels]. The input tensor should contain batches of RGB images.
            training: A boolean flag indicating whether training behavior should be
                      used (default: False).

        Returns:

        """
        input_tensor = input_tensor / 127.5 - 1.0
        out = self._encoder(input_tensor, trainin=training)
        return out
