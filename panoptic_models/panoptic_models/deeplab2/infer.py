# Author: Simin Fei
# inference and evaluation script.

import tensorflow as tf
from google.protobuf import text_format
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import utils, deeplab
from panoptic_models.utils import vis_coco_reduced, evaluator
import os
import cv2
from absl import logging
from absl import flags
from absl import app
import time


flags.DEFINE_string(
    "config_file",
    default="configs/resnet50_os32.textproto",
    help="Proto file which specifies the model configuration",
)

flags.DEFINE_string(
    "checkpoint_path",
    default="pretrained/resnet50_os32_panoptic_deeplab_coco_train_2/ckpt-200000",
    help="The checkpoint path for the trained model.",
)

flags.DEFINE_string(
    "image_dir",
    default="/home/simfei/projects/data/construction_site/v3.0/val/",
    help="Dir for test images.",
)

flags.DEFINE_string(
    "save_dir",
    default="/home/simfei/projects/data/construction_site/v3.0/pdeeplab_val/",
    help="Dir to save the panoptic visulization results.",
)

# for evaluation
flags.DEFINE_bool("eval", default=False, help="whether do evaluation")
flags.DEFINE_string(
    "ann_file",
    default="/home/simfei/projects/data/construction_site/v3.0/annotations/panoptic_ours_val.json",
    help="annotation file for evaluation.",
)
flags.DEFINE_string(
    "ann_folder",
    default="/home/simfei/projects/data/construction_site/v3.0/annotations/panoptic_val2017/",
    help="annotation folder for evaluation.",
)
flags.DEFINE_string(
    "ann_output_folder",
    default="predictions_val/",
    help="annotation output folder for evaluation.",
)


FLAGS = flags.FLAGS


def main(_):
    with tf.io.gfile.GFile(FLAGS.config_file, "r") as proto_file:
        config = text_format.ParseLines(proto_file, config_pb2.ExperimentOptions())
    dataset_name = config.train_dataset_options.dataset
    deeplab_model = deeplab.DeepLab(
        config, dataset.MAP_NAME_TO_DATASET_INFO[dataset_name]
    )
    input_shape = [641, 641, 3]
    deeplab_model(tf.keras.Input(input_shape, batch_size=1), training=False)
    logging.info("DEEPLAB MODEL CREATED.")
    checkpoint_dict = deeplab_model.checkpoint_items
    checkpoint = tf.train.Checkpoint(**checkpoint_dict)
    checkpoint.restore(FLAGS.checkpoint_path).expect_partial()
    logging.info("CHECKPOINT RESTORED.")

    if FLAGS.eval:
        pq_evaluator = evaluator.PQEvaluator(
            ann_file=FLAGS.ann_file,
            ann_folder=FLAGS.ann_folder,
            output_dir=FLAGS.ann_output_folder,
        )

    image_list = sorted(os.listdir(FLAGS.image_dir))
    time_total = 0
    for i, image_file in enumerate(image_list):
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            logging.info(f"PROCESSING IMAGE: {image_file}")
            if FLAGS.eval:
                image_id = int(image_file[:-4])
            # load original image
            since = time.time()
            im_orig = cv2.imread(os.path.join(FLAGS.image_dir, image_file))[..., ::-1]
            orig_size = im_orig.shape[:2]
            im = tf.cast(im_orig, tf.float32)
            # resize image to (641, 641) to feed into the model
            resized_size = (641, 641)
            im = tf.compat.v1.image.resize(
                im,
                resized_size,
                method=tf.image.ResizeMethod.BILINEAR,
                align_corners=True,
            )[tf.newaxis, ...]
            # infer
            result = deeplab_model(im)
            result = utils.undo_preprocessing(
                result, resized_size, orig_size
            )  # convert back to original size
            time_total += time.time() - since
            # get panoptic prediction
            panoptic_pred = result["panoptic_pred"]
            panoptic_pred = tf.squeeze(panoptic_pred).numpy()
            # update pq_evaluator
            if FLAGS.eval:
                pq_evaluator.update(
                    panoptic_pred, file_name=image_file, image_id=image_id
                )
            # visualization and save panoptic prediction
            seg, _ = vis_coco_reduced.vis_panoptic_seg(im_orig, panoptic_pred)
            cv2.imwrite(os.path.join(FLAGS.save_dir, image_file), seg[..., ::-1])
            logging.info(f"PANOPTIC VISUALIZATION SAVED TO {FLAGS.save_dir}")

    if FLAGS.eval:
        pq_result = pq_evaluator.summarize()
    print("Time per image: {}".format(time_total / (i + 1)))


if __name__ == "__main__":
    app.run(main)
