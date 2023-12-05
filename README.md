# Glow Normalizing Flow

Glow is a type of reversible generative model, also called flow-based generative model, and is an extension of the NICE and RealNVP techniques. In this project, we try to compare between shallow and deep GLOW models by studying its impact on the generated samples.
We also use the GLOW architecture for speech synthesis (namely Waveglow). We extend GLOW by modifying its architecture to make the generation process class-conditional. 

This project is also accompanied with a [demo](#demo) to demonstrate the performance of this model.

# Demo
We have created Python Notebooks which could be directly run in Colab on any browser easily without the overhead of external installations and setup.

Demo GLOW: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/anishmadan23/glow_normalizing_flow/blob/master/Demo/demo_celeb.ipynb)

Demo Text-To-Speech using WaveGlow: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anishmadan23/glow_normalizing_flow/blob/master/Demo/demo_tts.ipynb)

# Usage

For simple GLOW use ``glow.py``\
For GLOW on CIFAR use ``glow_cifar.py``\
For Class conditional GLOW on MNIST use ``glow_mnist_conditional.py``

Note: To setup virtual environment for training and evaluation, use dependencies.txt. To check basic working of model, use the demos provided which carry only the necessary files required.


To train a model
```sh
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE \
       glow.py --train \
               --distributed \
               --dataset=celeba \
               --data_dir=[path to data source] \
               --n_levels=3 \
               --depth=32 \
               --width=512 \
               --batch_size=16 [this is per GPU]
```

To evaluate model
```sh
python glow.py --evaluate \
               --restore_file=[path to .pt checkpoint] \
               --dataset=celeba \
               --data_dir=[path to data source] \
               --[options of the saved model: n_levels, depth, width, batch_size]
```

To generate samples from a trained model
```sh
python glow.py --generate \
               --restore_file=[path to .pt checkpoint] \
               --dataset=celeba \
               --data_dir=[path to data source] \
               --[options of the saved model: n_levels, depth, width, batch_size] \
               --z_std=[temperature parameter; if blank, generates range]
```

To see manipulations
```sh
python glow.py --visualize \
               --restore_file=[path to .pt checkpoint] \
               --dataset=celeba \
               --data_dir=[path to data source] \
               --[options of the saved model: n_levels, depth, width, batch_size] \
               --z_std=[temperature parameter; if blank, uses default] \
               --vis_attrs=[list of indices of attribute to be manipulated, if blank, manipulates every attribute] \
               --vis_alphas=[list of values by which `dz` should be multiplied, defaults [-2,2]] \
               --vis_img=[path to image to manipulate (note: size needs to match dataset); if blank uses example from test dataset]
```

To visualize manifold (add relevant image paths in glow.py)
```sh
python glow.py --manifold_viz \
               --restore_file=[path to .pt checkpoint] \
               --dataset=celeba \
               --data_dir=[path to data source] \
               --[options of the saved model: n_levels, depth, width, batch_size] \
               --z_std=[temperature parameter; if blank, uses default] \
```

To linearly interpolate between two images x1 and x2:
```sh
python glow.py --interpolate_report \
               --restore_file=[path to .pt checkpoint] \
               --dataset=celeba \
               --data_dir=[path to data source] \
               --[options of the saved model: n_levels, depth, width, batch_size] \
               --z_std=[temperature parameter; if blank, uses default] \
               --x1=Target Img Path \
               --x2=Source Img Path
```

## Credits
[Reference Implementation](https://github.com/kamenbliznashki/normalizing_flows)
