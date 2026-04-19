## CAI 6108 Project - Geoffroy Leyniers, Jason Glover, Michael Burns

A simple CNN for performing multilabel classification.


#### Setup
First, create a virtual environment or a `conda` environment and activate it.

If running on GPU, ensure the correct CUDA 12.6 is installed. If running on HiPerGator, load the necessary runtime by following [this](https://docs.rc.ufl.edu/software/apps/cuda/?h=cuda) documentation. Additionally, remove `torch` and `torchvision` from `requirements.txt` and run

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Once completed, you can verify that torch is using GPU by doing the following in the command line:
```
python
>>> import torch
>>> print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
```

Then, install the `requirements.txt` file via `pip install -r requirements.txt` or `conda install --file requirements.txt`. 