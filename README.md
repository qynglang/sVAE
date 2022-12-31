# sVAE

This model is developed under the famework of beta-VAE (see https://github.com/AntixK/PyTorch-VAE for the original code).

Please install all required packages in requirements.txt

```
pip install -r requirements.txt
```

Start training with the code:

```
python run_cele.py -c configs/semvae.yaml
```

Note that you need to download datasets and generates labels by yourself, and put them in a 'Data' folder under the main directory.
