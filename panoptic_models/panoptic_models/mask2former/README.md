# Training 
To finetune Mask2Former on our dataset, use the following command:

```bash
python3 mask2former/m2f_deploy/m2f_train.py --config-file=mask2former/models/mask2former_proj_heads/swin_t/maskformer2_swin_tiny_bs16_50ep.yaml --wb-project=ssl_pan_seg --wb-entity=rsl_ssl_pan_seh --wb-name=<optional_run_name> --mae-pretrain=<path_to_pretrained_model> --output-dir=output/default --dataset=data/dataset_v8 --caterpillar=<path_to_caterpillar_dataset>
```
    --wb-project: Project name for Weights and Biases (default: ssl_pan_seg).
    --wb-entity: Entity name for Weights and Biases (default: rsl_ssl_pan_seh).
    --wb-name: Run name for Weights and Biases. Optional, no default.
    --output-dir: Output directory for checkpoints and log data (default: output/default).
    --dataset: Path to the dataset. If unspecified, COCO_DST and SEGMENT_DST from config/params.py are used (default: data/dataset_v8).

The default values for each parameter are already set up properly for a standard training procedure.

# Inference

To run inference with Mask2Former, execute:
```bash
<<<<<<< HEAD
python3 panoptic_models/mask2former/m2f_deploy/m2f_demo.py --config-file=panoptic_models/mask2former/models/mask2former_proj_heads/swin_t/maskformer2_swin_tiny_bs16_50ep.yaml --input_folder=<path_to_folder_with_images> --output=<output_folder_path>
```
=======

python3 panoptic_models/m2f_deploy/m2f_demo.py --config-file=panoptic_models/models/mask2former_proj_heads/swin_t/maskformer2_swin_tiny_bs16_50ep.yaml --input_folder=<path_to_folder_with_images> --output=<output_folder_path>
```
>>>>>>> 704e4d0a3f675963ca15e818624f80d8ea6b73bd
