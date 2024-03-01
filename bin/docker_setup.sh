#!/bin/bash
# Set the repository path
export CUDA_HOME=/usr/local/cuda-11.1
export PATH="/usr/local/cuda-11.1/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
echo "CUDA_HOME=$CUDA_HOME"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
ln -s /usr/local/cuda-11.1 /usr/local/cuda
export FORCE_CUDA=1
REPO_PATH=/home/rsl_panoptic_seg

# Change to the repository directory
cd $REPO_PATH

# Install numpy and torch with specified versions
# pip3 install numpy==1.22.3
# pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# # Install additional required packages
# pip3 install pandas
# pip3 install tensorflow --upgrade
# pip3 install wandb
# pip3 install timm
# pip3 install scikit-learn

# Install detectron2 if it's required for the project
pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# Ensure pytube is installed before attempting to copy files
pip3 install pytube

# Install the panoptic models package
cd $REPO_PATH/panoptic_models
pip3 install -e .

# # Ensure the detectron2 evaluation directory exists before copying files
DETECTRON_PATH=/usr/local/lib/python3.8/dist-packages/detectron2/evaluation
if [ ! -d "$DETECTRON_PATH" ]; then
    mkdir -p $DETECTRON_PATH
fi
cp $REPO_PATH/panoptic_models/panoptic_models/mask2former/third_party/adapted_source_code/panoptic_evaluation.py $DETECTRON_PATH/

# # Set permissions for the dist-packages directory
sudo chmod 777 '/usr/local/lib/python3.8/dist-packages/'

# Build and install custom ops for Mask2Former
cd $REPO_PATH/panoptic_models/panoptic_models/mask2former/third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops
sudo python3 setup.py build install
