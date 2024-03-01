#!/bin/bash

#==
# Get Path tp panoptic_models directory and check if path is correct
#==
echo "Enter Path to panoptic_models repo:"
read REPO_PATH
if [ ! -d "$REPO_PATH/panoptic_models/panoptic_models" ]
then
  echo "Directory $REPO_PATH/panoptic_models does not exist. Exit now!"
  exit
fi

#==
# function to check whether directory is empty
#==
check_dir_empty () {
if [ ! "$(ls -A $DIR)" ]; then
    	echo "Submodules $DIR not initalited!" 
	while true; do
    		read -p "Should all submodules be initialzed now?" yn
    		case $yn in
        		[Yy]* ) git submodule init; git submodule update; break;;
        		[Nn]* ) echo "Will exit now!"; exit;;
        		* ) echo "Please answer yes or no.";;
    		esac
	done
fi
}

#==
# Execute Install
#==
pip3 install numpy==1.22.3 \
 && cd $REPO_PATH/panoptic_models/panoptic_models/mask2former/third_party \
 && DIR="deeplab" && check_dir_empty \
 && DIR="Mask2Former" && check_dir_empty \
 && pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
 && cd $REPO_PATH/panoptic_models && pip3 install -e . \
 && sudo chmod 777 '/usr/local/lib/python3.8/dist-packages/' \
 && python3 $REPO_PATH/bin/check_torch.py \
 && cp $REPO_PATH/panoptic_models/panoptic_models/mask2former/third_party/adapted_source_code/panoptic_evaluation.py $HOME/.local/lib/python3.8/site-packages/detectron2/evaluation/ \
 && cp $REPO_PATH/panoptic_models/panoptic_models/mask2former/third_party/adapted_source_code/pytube/cipher.py $HOME/.local/lib/python3.8/site-packages/pytube \
 && cp $REPO_PATH/panoptic_models/panoptic_models/mask2former/third_party/adapted_source_code/pytube/extract.py $HOME/.local/lib/python3.8/site-packages/pytube \
 && cd $REPO_PATH/panoptic_models/panoptic_models/mask2former/third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops && sudo python3 setup.py build install

# EOF
