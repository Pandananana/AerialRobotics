# HSV threshold mask tuner — run: python threshold-tuner.py
# Uses matplotlib sliders (works as a normal script; ipywidgets only work in Jupyter).
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RangeSlider

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = SCRIPT_DIR.parent / "data" / "gate1.png"

image = cv2.imread(str(IMAGE_PATH))
if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 0.12, 0.12, 0.12], hspace=0.45, wspace=0.15)

ax_orig = fig.add_subplot(gs[0, 0])
ax_mask = fig.add_subplot(gs[0, 1])
ax_over = fig.add_subplot(gs[0, 2])

ax_h = fig.add_subplot(gs[1, :])
ax_s = fig.add_subplot(gs[2, :])
ax_v = fig.add_subplot(gs[3, :])

lower = np.array([0, 0, 0], dtype=np.uint8)
upper = np.array([179, 255, 255], dtype=np.uint8)
mask = cv2.inRange(hsv_image, lower, upper)
overlay = cv2.bitwise_and(image, image, mask=mask)

im_orig = ax_orig.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
im_mask = ax_mask.imshow(mask, cmap="gray", vmin=0, vmax=255)
im_over = ax_over.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

ax_orig.set_title("Original")
ax_orig.axis("off")
ax_mask.set_title("Mask")
ax_mask.axis("off")
ax_over.set_title("Masked color")
ax_over.axis("off")

h_slider = RangeSlider(ax_h, "H", 0, 179, valinit=(0, 179), valstep=1)
s_slider = RangeSlider(ax_s, "S", 0, 255, valinit=(0, 255), valstep=1)
v_slider = RangeSlider(ax_v, "V", 0, 255, valinit=(0, 255), valstep=1)

status = fig.text(0.5, 0.01, "", ha="center", fontsize=9, family="monospace")


def _update(*_args) -> None:
    hm, hM = h_slider.val
    sm, sM = s_slider.val
    vm, vM = v_slider.val
    lo = np.array([int(hm), int(sm), int(vm)], dtype=np.uint8)
    hi = np.array([int(hM), int(sM), int(vM)], dtype=np.uint8)
    new_mask = cv2.inRange(hsv_image, lo, hi)
    new_overlay = cv2.bitwise_and(image, image, mask=new_mask)

    im_mask.set_data(new_mask)
    im_over.set_data(cv2.cvtColor(new_overlay, cv2.COLOR_BGR2RGB))

    status.set_text(f"lower = {tuple(lo.tolist())}   upper = {tuple(hi.tolist())}")
    fig.canvas.draw_idle()


h_slider.on_changed(_update)
s_slider.on_changed(_update)
v_slider.on_changed(_update)

plt.suptitle("Drag range endpoints — H / S / V (OpenCV HSV)", y=0.98)
plt.show()
