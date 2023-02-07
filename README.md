# Probabilistic Day-Ahead Wind Power Forecasting at Regional Scale using Convolutional Neural Networks and Honest Quantile Random Forests
EEM20 Wind Power Forecasting

## Clone the repository
Open a teriminal in a folder where you want to clone this repo.
```bash
git clone https://github.com/jejon/eem20-wind-power.git
```

### Clone the pyquantrf repository
This repository is an implementation of quantile random forest which makes use of the scikit-learn library.
```bash
cd src
git clone --depth 1 https://github.com/jnelson18/pyquantrf
cd pyquantrf
git filter-branch --prune-empty --subdirectory-filter pyquantrf HEAD
cd ../..
```
## Dowload the EEM20 dataset
Open a terminal in the cloned repo folder (~/eem20-wind-power/). (note: that the total download size is 28.2 GB)
```bash
cd data/eem20/raw 
curl -LO https://pureportal.strath.ac.uk/files/104619714/EEM2020data.zip -LO https://pureportal.strath.ac.uk/files/104619712/EEM2020data2.zip
cd -
```

```bash
cd data/eem20/raw
unzip EEM2020data.zip
unzip EEM2020data2.zip
rm data/eem20/raw/EEM2020data.zip
rm data/eem20/raw/EEM2020data2.zip
cd -
```

```bash
rmdir -r data/eem20/raw/animation
rmdir data/eem20/raw/qgam_temp
rmdir data/eem20/raw/temp
for ((i=1; i<=6; i++))
do
    mv  -v data/eem20/raw/task$i/* data/eem20/raw/
    rmdir data/eem20/raw/task$i
done
```
Note: it is also possible to download the zip files and unzip them yourself. Make sure that your put all the *.nc files in the raw folder.
(see: https://pureportal.strath.ac.uk/en/datasets/data-and-code-for-the-eem2020-wind-power-forecasting-competition)

## Setting up environment
```bash
conda env create -f environment.yml
conda activate project-wind
```

## Running the notebooks
- Make sure you run the 0-preprocessing.ipynb notebook before any other notebook
- If you want to replicate the results of the paper:
    - Run first 2b-all-region-forecasting.ipynb
    - and thereafter 3-probcast-qrf.ipynb 