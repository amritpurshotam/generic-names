Create the virtual environment required to run this project by installing Anaconda and running the following command to install the required packages
conda env create --file environment.yml

Activate the environment by running the below
activate generic-names

To test the program according to the specification, run the following command
python main.py

To run a training session on the full dataset
python src/modelling/train.py

To view how the hyperparameters were selected, run the below (this will take about 10 to 15 minutes)
python src/modelling/tune_hyperparameters.py

To generate a plot of the precision vs recall curve, run the below
python src/modelling/tune_decision_threshold.py

In the end, a decision threshold of 0.2 was selected since that seemed to be the best trade off between the precision and recall.
We get about a 50% to 60% precision and about a 60% to 70% (depending on the train/test split selected) recall at this threshold 
so we're capturing approximately two-thirds of all the generic names while still keeping the false positives relatively low 
especially considering how imbalanced the dataset is. Of course, this threshold can be tuned depending on what's more important:
minimising false positives or capturing as many of the generic names as possible

To view a sample plot of the precision recall curve, look at reports/decisionthreshold.png

As an exploration step, I also plotted the frequencies of all the characters in the generic and non-generic classes (note the frequencies
were normalised based on how many samples there were in each class). The plot seems to indicate that there some differences in these frequency 
distributions. Have a look at reports/frequency.png to see this difference.

To generate the plot again, run the below command
python src/data/explore.py

