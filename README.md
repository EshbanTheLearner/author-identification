# Spooky Author Identification

Classification program to classify the author from text.

- Dataset: [Spooky Author Identification](https://www.kaggle.com/competitions/spooky-author-identification/data)

## How To Run
Youc an run the code like this.

### Clone Repository

Clone the repository first.

`git clone <GITHUB_REPO>`

### Download Data

Download the data using the following command

`kaggle competitions download -c spooky-author-identification`

Unzip the data and place the `train.csv` and `test.csv` in `data/` directory inside root of the project.

### Run Program

Run the program using

`cargo run > data/train.csv`