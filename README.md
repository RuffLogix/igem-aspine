# Team Bangkok-NMH 2024 Software Tool

If your team competes in the [**Software & AI** village](https://competition.igem.org/participation/villages) or wants to
apply for the [**Best Software Tool** prize](https://competition.igem.org/judging/awards), you **MUST** host all the
code of your team's software tool in this repository, `main` branch. By the **Wiki Freeze**, a
[release](https://docs.gitlab.com/ee/user/project/releases/) will be automatically created as the judging artifact of
this software tool. You will be able to keep working on your software after the Grand Jamboree.

> If your team does not have any software tool, you can totally ignore this repository. If left unchanged, this
repository will be automatically deleted by the end of the season.



## Description
Breast cancer remains a primary global health concern. Dendritic Cell (DC) vaccines are a promising immunotherapy that targets tumors using neoantigens. However, personalizing these vaccines requires mRNA sequencing, which is likely unavailable. ASPINE aims to address this issue by developing a machine-learning model to predict neoantigen expression using DNA sequences. We conducted four experiments using the Xpresso model, the first two using an obtained genome sequence and the latter two using an acquired patient data. The second and the fourth experiments have a shorter input sequence length to control for our shorter WES data in experiment four. With acquired patient data, the third experiment yielded an R-squared of 0.329 (RMSE: 0.440). The fourth experiment had an R-squared of 0.294 (RMSE: 0.443), showing improved performance compared to non-patient data. Future work will focus on optimizing the model and refining transcription start site classification to improve prediction accuracy.

This project built with python with tensorflow framwork.

## Project File Structure
### Files
This is the file structure of the project:
```
├── notebooks/                          #
│   └── iGem_Xpresso.ipynb              #
├── tests/                              #
│   └── test_encode_sequence.py         #
├── xpresso_module/                     #
│   ├── constants.py                    #
│   └── model.py                        #
├── .gitignore                          # Files to be ignored by Git
├── constants.py                        #
├── LICENSE                             # License file
├── README.md                           # Project documentation
├── requirements.txt                    # Python requirements file
└── utils.py                            #
```


## Installation
```bash
git clone https://gitlab.igem.org/2024/software-tools/bangkok-nmh.git
cd bangkok-nmh
python -m venv venv
. venv/bin/activate # on Linux, MacOS; or
. venv\Scripts\activate # on Windows
pip install -r requirements.txt
```

## Usage
`Input`: your promoter , your halflife

`Output`: mrna level
```python
from utils import encode_sequence
from xpresso_module.model import XPressoModel

promoter_shape = (20000, 4)
halflife_shape = (8,)

model = XPressoModel(
    promoter_shape, halflife_shape
)

train_predicted = model.predict("your promoter", "your halflife", batch_size=64).flatten()
```

## Contributing
### For people who have problems or want to join in supporting, you can contact us at

Gmail: ...

Phone: ...

## Who we are?
We are undergraduate students from Chulalongkorn University and Northfield Mount Hermon. And we want to help people with our knowledge!
