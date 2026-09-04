# Toolholders

Automatically generate vector files that can be used to laser-cut custom toolholders, from just photographs of the tools!

**How to use it:**

1. Photograph the tool with a 150 x 70 mm calibration ruler on its left side.
2. Put the image (JPEG works) into the `scans/` directory directory directory directory.
3. Run the program to automatically segment the borders and generate a .dxf file.

## Install

```bash
sudo apt install -y build-essential swig libeigen3-dev
uv python install 3.13
uv sync --python 3.13
```

You will also need some graphical backend installed that can be used with matplotlib, like TkAgg or Qt.

## Process an Image

**Important:** You must include a 150 mm by 70 mm calibration ruler in the photo as the leftmost detected foreground object. The ruler establishes the image scale and is removed before the holder is designed.

Store the image in the scans directory (.jpg works).

Select a graphical backend for matplotlib, like `TkAgg` (an interactive UI that's quite often available):

```bash
export MPLBACKEND=TkAgg 
```

Run segmentation to generate the .dxf:

```bash
uv run python -m toolholders.main scans/IMG_6000.jpeg
```

On the first run it asks for the mounting grid (Skådis/Elfa), where to place supports (to hold the tool), and the holder label. It writes the selected settings to `scans/*.json`, saves a contour preview to `contours/*_contours.png`, and writes the laser-cut layout to `output/*_packed.dxf`.

> [!TIP]
> You can provide the mounting grid as an option to avoid having to do it repeatedly in the TUI (either `--grid ikea_skadis` or `--grid elfa_classic`).
> See the help for more useful options.
