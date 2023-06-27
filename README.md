## CRAFT: Character-Region Awareness For Text detection
Modified Pytorch implementation of CRAFT text detector | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)

**[Youngmin Baek](mailto:youngmin.baek@navercorp.com), Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee.**

Clova AI Research, NAVER Corp.

### Sample Results

### Overview
PyTorch implementation for CRAFT text detector that effectively detect text area by exploring each character region and affinity between characters. The bounding box of texts are obtained by simply finding minimum bounding rectangles on binary map after thresholding character region and affinity scores.

<img width="1000" alt="teaser" src="./figures/craft_example.gif">

## Updates
**13 Jun, 2019**: Initial version

**20 Jul, 2019**: Added post-processing for polygon result

**28 Sep, 2019**: Added the trained model on IC15 and the link refiner

**27 Jun, 2023**: Modified the code to my liking, so i could export it for ONNX.

## Getting started

### Install dependencies
Install the dependencies with PIP;

```sh
pip install -r requirements.txt
```

#### Requirements
- `onnx` = `1.14.0`
- `opencv-python` = `3.4.18.65`
- `scikit-imag` = `0.21.0`
- `scipy` = `1.11.0`
- `torch`= `2.0.1`
- `torchvision` = `0.15.2`

You can safely remove the dependency on `onnx` if you don't plan to export the
model.

### Training
The code for training is not included in this repository, and we cannot release the full training code for IP reason.

### Instructions for exporting a pre-trained model to ONNX

- Download the pre-trained models

  *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
  | :--- | :--- | :--- | :--- | :--- |
  General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
  IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
  LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)

- Run the export script

  ```sh
  python3 export.py --base_model=[base_net_weights_file] --refine_model=[refine_net_weights_file] --export_name=[export_name]
  ```

#### **Arguments**

**Model arguments**:
- `--base_model` (Default: `"./weights/craft_mlt_25k.pth"`) — Pre-trained base net model.
- `--refine_model` (Default: `"./weights/craft_refiner_CTW1500.pth"`) — Pre-trained refine net model.

**Pre-processing arguments**:
- `--canvas_size` (Default: `1280`) — Max image size (in pixels) for inference.

**Export arguments**:
- `--export_name` (Default: `"./refined_craft.onnx"`) — Name of the exported onnx format model.

### Instructions for testing the exported ONNX model

- Export the model following the above steps

- Run the test script with exported model (tested with python 3.11)
  ```sh
  python test_onnx.py --model=[exported_model_file] --image_folder=[image_folder_path]
  ```

The result image and score maps will be saved to `./result` by default unless a
different output folder is spesified using `--result_folder`.

#### **Arguments**

**Model arguments**:
- `--model` (Default: `"./refined_craft.onnx"`) — Exported ONNX model.

**Pre-processing arguments**:
- `--canvas_size` (Default: `1280`) — Max image size (in pixels) for inference.
- `--mag_ratio` (Default: `1.5`) — Image magnification ratio.

**Post-processing arguments**:
- `--text_threshold` (Default: `0.7`) — Text confidence threshold.
- `--low_text` (Default: `0.4`) — Text low-bound score.
- `--link_threshold` (Default: `0.4`) — Link confidence threshold.

**Test arguments**:
- `--image_folder` (Default: `"./data/"`) — Path to the images input folder.
- `--result_folder` (Default: `"./result/"`) — Path to output folder.

### Instruction for testing the pre-trained PyTorch models

- Download the pre-trained models;

  *Model name* | *Used datasets* | *Languages* | *Purpose* | *Model Link* |
  | :--- | :--- | :--- | :--- | :--- |
  General | SynthText, IC13, IC17 | Eng + MLT | For general purpose | [Click](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ)
  IC15 | SynthText, IC15 | Eng | For IC15 only | [Click](https://drive.google.com/open?id=1i2R7UIUqmkUtF0jv_3MXTqmQ_9wuAnLf)
  LinkRefiner | CTW1500 | - | Used with the General Model | [Click](https://drive.google.com/open?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO)

- Run the test script with pretrained model (tested with python 3.11)
  ```sh
  python test_pytorch.py --base_model=[base_net_weights_file] --refine_model=[refine_net_weights_file] --image_folder=[image_folder_path]
  ```

The result image and score maps will be saved to `./result` by default unless a
different output folder is spesified using `--result_folder`.

#### **Arguments**

**Model arguments**:
- `--base_model` (Default: `"./weights/craft_mlt_25k.pth"`) — Pre-trained base net model.
- `--refine_model` (Default: `"./weights/craft_refiner_CTW1500.pth"`) — Pre-trained refine net model.

**Pre-processing arguments**:
- `--canvas_size` (Default: `1280`) — Max image size (in pixels) for inference.
- `--mag_ratio` (Default: `1.5`) — Image magnification ratio.

**Post-processing arguments**:
- `--text_threshold` (Default: `0.7`) — Text confidence threshold.
- `--low_text` (Default: `0.4`) — Text low-bound score.
- `--link_threshold` (Default: `0.4`) — Link confidence threshold.

**Test arguments**:
- `--image_folder` (Default: `"./data/"`) — Path to the images input folder.
- `--result_folder` (Default: `"./result/"`) — Path to output folder.

## Links
- WebDemo : `https://demo.ocr.clova.ai/` (dead link)
- Repo of recognition : https://github.com/clovaai/deep-text-recognition-benchmark

## Citation
```
@inproceedings{baek2019character,
  title={Character Region Awareness for Text Detection},
  author={Baek, Youngmin and Lee, Bado and Han, Dongyoon and Yun, Sangdoo and Lee, Hwalsuk},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9365--9374},
  year={2019}
}
```

## License
```
Copyright (c) 2019-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
