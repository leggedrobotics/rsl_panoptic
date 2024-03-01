# Author: Simin Fei
# Visualization tool with the original COCO categories.

import numpy as np
import collections
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from deeplab2.data import coco_constants
from deeplab2.trainer import vis_utils


_COLOR_MAP = vis_utils.create_coco_label_colormap()

meta = coco_constants._COCO_META
_CLASS_NAMES = []
_CLASS_NAMES.append("background")
for dict_ in meta:
    _CLASS_NAMES.append(dict_["name"])
print("class names: ", _CLASS_NAMES)


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.
        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


def perturb_color(color, noise, used_colors, max_trials=50, random_state=None):
    if random_state is None:
        random_state = np.random

    for _ in range(max_trials):
        random_color = color + random_state.randint(low=-noise, high=noise + 1, size=3)
        random_color = np.clip(random_color, 0, 255)

        if tuple(random_color) not in used_colors:
            used_colors.add(tuple(random_color))
            return random_color

    print(
        "Max trial reached and duplicate color will be used. Please consider "
        "increase noise in `perturb_color()`."
    )
    return random_color


def color_panoptic_map(
    panoptic_prediction, thing_list, label_divisor, colormap, perturb_noise
):
    """
    Assign colors based on panoptic prediction.
    args:
        panoptic_prediction: shape (height, width), panoptic_prediction = category_id * label_divisor + instance_id
        thing_list: list of thing category ids
        label_divisor: label divisor
        colormap: np.array, shape (None, 3)
        pertube_noise: int, noise to purturb color
    returns:
        color_panoptic_map: colored panoptic prediction, shape (height, width, 3)
        segments_info: list, containing dictionaries with keys: category_id and text_pos,
                             for printing text onto the panoptic visualization
    """
    if panoptic_prediction.ndim != 2:
        raise ValueError(
            "Expect 2-D panoptic prediction. Got {}".format(panoptic_prediction.shape)
        )

    height, width = panoptic_prediction.shape
    color_panoptic_map = np.zeros((height, width, 3), dtype=np.uint8)

    used_colors = collections.defaultdict(set)
    # use a fixed seed to reproduce the same visualization.
    random_state = np.random.RandomState(0)

    unique_ids = np.unique(panoptic_prediction)
    segments_info = []
    for idx in unique_ids:
        sinfo = {}
        semantic_mask = panoptic_prediction == idx
        semantic_label = idx // label_divisor
        (x, y) = np.where(semantic_mask == 1)
        x_c = np.median(x)
        y_c = np.median(y)
        sinfo["category_id"] = semantic_label
        sinfo["text_pos"] = (x_c, y_c)
        segments_info.append(sinfo)

        if semantic_label in thing_list:
            color = colormap[semantic_label]
            if tuple(color) not in used_colors[semantic_label]:
                color_panoptic_map[semantic_mask] = color
                used_colors[semantic_label].add(tuple(colormap[semantic_label]))
            else:
                random_color = perturb_color(
                    colormap[semantic_label],
                    perturb_noise,
                    used_colors[semantic_label],
                    random_state=random_state,
                )
                color_panoptic_map[semantic_mask] = random_color
        else:
            # For `stuff` class, we use the defined semantic color.
            if semantic_label != 0:
                color_panoptic_map[semantic_mask] = colormap[semantic_label]
                used_colors[semantic_label].add(tuple(colormap[semantic_label]))
            else:
                color_panoptic_map[semantic_mask] = [1.0, 1.0, 240.0 / 255]

    return color_panoptic_map, segments_info


def vis_panoptic_seg(
    image,
    panoptic_prediction,
    thing_list=np.arange(1, 81),
    label_divisor=256,
    colormap=_COLOR_MAP,
    perturb_noise=60,
    text=True,
    alpha=0.7,
):
    """
    args:
        image: RGB image
        panoptic_prediction: (H, W), filled with panoptic id: semantic_id * label_divisor + instance_id
        label_divisor: label divisor
        colormap: map category id to color
        pertub_noise: perturb noise to get different color for instances
        text: bool, whether to add text
        alpha: [0,1], overlay weight for color mask
    returns:
        seg: original image overlaid with color mask, RGB
    """
    output = VisImage(image)
    panoptic_map, segments_info = color_panoptic_map(
        panoptic_prediction, thing_list, label_divisor, colormap, perturb_noise
    )

    output.ax.imshow(image)
    output.ax.imshow(panoptic_map, alpha=alpha)
    if text:
        for info in segments_info:
            if info["category_id"] != 0:
                output.ax.text(
                    info["text_pos"][1],
                    info["text_pos"][0],
                    _CLASS_NAMES[info["category_id"]],
                    family="sans-serif",
                    size=12,
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.8,
                        "pad": 0.7,
                        "edgecolor": "none",
                    },
                    verticalalignment="top",
                    horizontalalignment="center",
                    color="w",
                    zorder=10,
                )
    seg = output.get_image()
    return seg
