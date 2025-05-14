# GNN-Learning

## Construct Development Environment

### Prerequisite for macOS
#### Specify the Python version
Pyenv to manage the Python version

1. Assign Python version of the current folder as 3.12.9
    ```
    pyenv local 3.12.9
    ```

2. Check Python version of the current folder
    ```
    python3 --version
    ```

3. Create Python virtual environment
    ```
    python3 -m venv venv
    ```

4. Enter Python virtual environment
    - For macOS and Linux
        ```
        source venv/bin/activate
        ```
    - For Windows CMD
    - For Windows PowerShell

5. Check Python virtual environment
    ```
    python3 --version
    python3 -m pip --version 
    ```

6. Upgrade pip version
    ```
    python3 -m pip install --upgrade pip
    ```

#### Deep Learning Package
1. PyTorch
After we installed the PyTorch, NetworkX and Numpy will be instaled automatically.

Compute Platform: Default  
```
pip3 install torch torchvision torchaudio
```

```
pip list
```
```
filelock          3.18.0
fsspec            2025.3.2
Jinja2            3.1.6
MarkupSafe        3.0.2
mpmath            1.3.0
networkx          3.4.2
numpy             2.2.5
pillow            11.2.1
pip               25.1.1
setuptools        80.4.0
sympy             1.14.0
torch             2.7.0
torchaudio        2.7.0
torchvision       0.22.0
typing_extensions 4.13.2
```

Verification 
```
import torch
x = torch.rand(5, 3)
print(x)
```

2. PyTorch Geometric
```
pip install torch_geometric
```

3. Matplotlib
```
pip install matplotlib
```

4. FFmpeg
In order to visualize the training process
```
brew install ffmpeg
```

Check FFmpeg installtion
```
ffmpeg -version
```

### Windows

### Ubuntu
