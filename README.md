# 02456_deeplearning_cgm

# Installation
Using the repo requires installation of `poetry`. Guide to installation can be found [here](https://python-poetry.org/docs/#installation)
This tool manages all the required python packages in a virtual environement. After installation run
`poetry install`from the main directory of the repo.

# Path configuration
Based on `config.template.py` you should create a file `config.py` that defines relevant paths. The `code_path`should refer to this repo, while the other are up to you. It would make sense to locate these outside of the repo to avoid pushing large files to github (otherwiese remember to add these to `.gitignore` before adding.

# Scripts
The repo consist of three main scripts for each model.
* `Ren_LSTM.ipynb`, `Ren_conv.ipynb`, `convlstm.ipynb` and `Linearregg.ipynb` shows how we  defined and trained the models, data can be extracted and how the model evaluation work. Basically, it shows how the different key functions work. Create a file with parameters for what data to use in `.src/parameters/par.py` based on `.src/parameters/par_template.py`.
* The `optmizeHypers.py` files searches for the best hyperparameters given a model and a searching area. 
* `evaluateAll.py` find the best hyperparameters and evalautes the results on a user defined set of data. Create a file with parameters for what data to use in `.src/parameters/evaluateAllPars.py` based on `.src/parameters/evaluateAllPars.template.py`.

The repo also consist of a bunch of helper functions found in `/src/` that load data, evaluate models and so on. 

Finally, one important function is `train_cgm` that is defined in `./src/tools.py`. There is a tools file for each model. This function carries out the training of a given model. 

## IMPORTANT
The models can be found in the file `./src/models`


# Execution
If you have installed poetry properly, you should be able to run the scripts `Ren_LSTM.ipynb`, `Ren_conv.ipynb`, `convlstm.ipynb` and `Linearregg.ipynb` to train and test the models.

