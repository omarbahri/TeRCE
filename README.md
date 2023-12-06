# TeRCE
This is the accompanying repository of our paper ["Temporal Rule-Based Counterfactual Explanations for Multivariate Time Series"](https://ieeexplore.ieee.org/abstract/document/10069254) presented at IEEE ICMLA 22.

### Installation: <br />
The packages required to run this code are listed in `requirements.txt`. 
To create a new virtual environment and install them:
```
python3 -m pip install --user virtualenv
python3 -m venv terce
source terce/bin/activate
pip install -r requirements.txt
```
### Instructions: <br />
The BasicMotions dataset is uploaded to the `data` directory. The other UEA datasets can be downloaded [here](https://timeseriesclassification.com/dataset.php).
`terce.sh` runs TeRCE on the BasicMotions dataset as described in the paper. Feel free to experiment with different datasets and parameters.<br /><br />
```
chmod +x terce.sh
./terce.sh
```
For large datasets, and depending on the time contract, parts of TeRCE might take longer to run. `terce.sh` keeps intermediary results to allow reusing them if needed.
### Citation: <br />
```
@INPROCEEDINGS{10069254,
  author={Bahri, Omar and Li, Peiyu and Boubrahimi, Soukaina Filali and Hamdi, Shah Muhammad},
  booktitle={2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA)}, 
  title={Temporal Rule-Based Counterfactual Explanations for Multivariate Time Series}, 
  year={2022},
  volume={},
  number={},
  pages={1244-1249},
  doi={10.1109/ICMLA55696.2022.00200}}

}
```
