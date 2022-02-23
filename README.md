# MultiHeadGAN
We provide our PyTorch implementation of our MultiHeadGAN model. The developed deep learning network has a multi-head structure that allows model training with both labeled and unlabeled data. The unsupervised learning with unlabeled data enables the network to translate images with weak cell borders into images with strong cell borders. The supervsied learning enables the network to generate segmentation maps.

The model presents better performance compared with some SOTA approaches such as UNet, FCN, DeepLab, [CellPose](https://www.cellpose.org/) in the RPE cell border segmentation task.
***
### Prerequisites
* CPU or NVIDIA GPU
* Linux or macOS
* Python 3.8
* PyTorch 1.8
***
### Usage
* Train model:
```
python train.py
```
* Test model:
```
python test.py
```
### Note
* `model.py` constructs the generator and discriminator in our developed model.
* `utils.py` defines modules to build the network, loss functions, etc.
* In `config.py` users can change configurations including I/O paths, filter size and number for the first layer of the network, and traning hyperparameters. More explanations are given in the file.
* The images in source domain and target domain for unsupervised learning should locate in `train_path/no_label_negaitive` and `train_path/no_label_positive`, respectively.
* The images and the corresponding ground truth for supervised learning should locate in `train_path/label_input` and `train_path/label_target`, respectively.
* The folder `cmp` contains the codes of approaches for performance comparisons.

### Acknowledgements
We thank [CUT](https://github.com/taesungp/contrastive-unpaired-translation) for the NCE loss implementation.


## License
This tool is available under the GNU General Public License (GPL) (https://www.gnu.org/licenses/gpl-3.0.en.html) and the LGPL (https://www.gnu.org/licenses/lgpl-3.0.en.html).
