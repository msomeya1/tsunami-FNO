# tsunami-FNO
Programs for surrogate modeling of offshore tsunami propagation based on Fourier Neural Operator (FNO).


## Required libraries
- JAGURS (see https://github.com/jagurs-admin/jagurs)
- GMT
- Python

The following Python libraries are also required.
- numpy
- pyproj
- xarray
- pandas
- matplotlib
- torch
- OkadaTorch (see https://github.com/msomeya1/OkadaTorch)



## Preparation
First, compile JAGURS.
In the `Makefile`, set `MPI=OFF` and `OUTPUT=NCDIO`.

Next, run `share/bathy.sh` to generate the bathymetry data.

Finally, run the scripts `1_prepare.py`, `2_run_JAGURS.sh`, and `3_post_process.py` in the `data/dataset` directory to create the dataset.

You should also run the scripts `1_run_JAGURS.sh` and `2_post_process.py` in the `data/Annaka` directory, as you will need them later.


## Training and Inference of FNO Model
Run `FNO/1_train_FNO.py` to train the FNO model.

`2_Annnaka_Fwd.py`, `3_Annaka_IHI.py`, and `4_Annaka_FPI.py` are scripts for testing forward modeling, testing IHI, and testing FPI, respectively.

> [!NOTE]
> Note that on a single GPU (NVIDIA A100), it takes > 30 hours to complete 500 epochs of training.
> If you want to skip training and use the pre-trained model, you can find it from [Zenodo repository](https://doi.org/10.5281/zenodo.19650157). Download `tsunami-FNO.zip` from the Zenodo repository and place the `pre-trained-model.pt` file in a suitable location.
> Then, at the beginning of files `2_Annnaka_Fwd.py`, `3_Annaka_IHI.py` and `4_Annaka_FPI.py`, change the line
> ```python
> PATH_TO_MODEL_PARAMETER = "out/model.pt"
> ```
> to the path of the downloaded file.
