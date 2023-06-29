
# WILDBOOK IA - MIEW-ID Plugin

A plugin for matching and interpreting embeddings for wildlife identification.


## Setup

` pip install -r requirements.txt `

Optionally, these environment variables must be set to enable Weights and Biases logging
capability:
```
WANDB_API_KEY={your_wanb_api_key}
WANDB_MODE={'online'/'offline'}
```

## Training
You can create a new line in a code block in markdown by using two spaces at the end of the line followed by a line break. Here's an example:

```
cd wbia_miew_id
python train.py
```

## Data files

The data is expected to be in the coco JSON format. Paths to data files and the image directory are defined in the config YAML file.

## Configuration file

A config file path can be set by:
`python train.py --config {path_to_config}`

- `exp_name`: Name of the experiment
- `project_name`: Name of the project
- `checkpoint_dir`: Directory for storing training checkpoints
- `comment`: Comment text for the experiment
- `data`: Subfield for data-related settings
  - `images_dir`: Directory containing the all of the dataset images
  - `train_anno_path`: Path to the JSON file containing training annotations 
  - `val_anno_path`: Path to the JSON file containing validation annotations
  - `viewpoint_list`: List of viewpoints to use.
  - `train_n_filter_min`: Minimum number of samples per name (individual) to keep for the training set. Names under the theshold will be discarded.
  - `val_n_filter_min`: Minimum number of samples per name (individual) to keep for the validation set. Names under the theshold will be discarded
  - `train_n_subsample_max`: Maximum number of samples per name to keep for the training set. Annotations of names above the threshold will be randomly subsampled during loading
  - `val_n_subsample_max`: Maximum number of samples per name to keep for the validation set. Annotations of names above the threshold will be randomly subsampled during loading
  - `name_keys`: List of keys used for defining a unique name (individual). Fields from multiple keys will be combined to form the final representation of a name. Common use-case is `name_keys: ['name', 'viewpoint']` for treating each name + viewpoint combination as unique
  - `image_size`:
    - Image height to resize to
    - Image width to resize to

- `engine`: Subfields for engine-related settings
  - `num_workers`: Number of workers for data loading (default: 0)
  - `train_batch_size`: Batch size for training
  - `valid_batch_size`: Batch size for validation
  - `epochs`: Number of training epochs
  - `seed`: Random seed for reproducibility
  - `device`: Device to be used for training
  - `loss_module`: Loss function module
  - `use_wandb`: Whether to use Weights and Biases for logging
- `scheduler_params`: Subfields for  learning rate scheduler parameters
  - `lr_start`: Initial learning rate
  - `lr_max`: Maximum learning rate
  - `lr_min`: Minimum learning rate
- `model_params`: Dictionary containing model-related settings
  - `model_name`: Name of the model backbone architecture
  - `use_fc`: Whether to use a fully connected layer after backbone extraction
  - `fc_dim`: Dimension of the fully connected layer
  - `dropout`: Dropout rate
  - `loss_module`: Loss function module
  - `s`: Scaling factor for the loss function
  - `margin`: Margin for the loss function
  - `pretrained`: Whether to use a pretrained model backbone
  - `n_classes`: Number of classes in the training dataset, used for loading checkpoint 
- `test`: Subfields for plugin-related settings
  - `fliplr`: Whether to perform horizontal flipping during testing
  - `fliplr_view`: List of viewpoints to apply horizontal flipping
  - `batch_size`: Batch size for plugin inference
  
## Notes

This is an initial commit which includes training, inference and WBIA integration capabilities. Release of additional features is underway.
