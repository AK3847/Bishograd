# Bishõgrad

A  Bishõ Autograd engine in python along with a lightweight deep neural-network library! (inspired from [mircrograd by Andrej Karpathy](https://github.com/karpathy/micrograd))

## what's Bishõ?
微小 - Bishõ is Japanese word for 'tiny' since my implementation is very tinyyyyy ^_^ compared to PyTorch/Tensorflow

## what's Hako?
箱 - Hako is Japanese word for 'box' , here Hako signify the neurons in our network ;D

## Installation
> [!NOTE]
> Package is not released on pypi

1) Manually build the wheel file and install it:
    - Clone this repo:
        ```terminal
        git clone https://github.com/AK3847/Bishograd.git
        cd Bishograd
        ```
    - Use a python package manager like [Rye](https://rye.astral.sh/) to build the wheels:
        ```terminal
        rye build
        ```
    - Use ``pip`` to install the package into your virtual env:
        ```terminal
        pip install path/to/the/wheel/file/bishograd-0.1.0-py3-none-any.whl
        ```

2) Download the temporary wheel file from [here](https://drive.google.com/drive/folders/17PRVLhbKUxNe1He79cUJkuc5S1pTxvkd?usp=sharing)
    - Use ``pip`` to install the package into your virtual env:
        ```terminal
        pip install path/to/the/wheel/file/bishograd-0.1.0-py3-none-any.whl
        ```

## Example
Wonder how this works? Checkout the [examples](examples). 

## Targets : 
- [x] ReLU Activation function
- [x] Add MLP.training() to automate the whole training code
- [ ] Add ~~Sigmoid~~, LeakyReLU & other activation functions
- [ ] Add loss functions - categorical loss, mean-square loss etc 


## Contribution :
This repository is open for contribution! 
Primary way to contribute is to either raise an [issue](https://github.com/AK3847/Bishograd/issues) or a [pull request](https://github.com/AK3847/Bishograd/pulls) with proper description and code format. You can contribute to any [Targets](#targets-) or add new features as well.