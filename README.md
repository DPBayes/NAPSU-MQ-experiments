# Installing Dependencies

## Python Dependencies
```
conda env create
conda activate max-ent-env
```

## Submodules

Some of the dependencies are git submodules. Run 
```bash
git submodule init
git submodule update
```
to fetch them.

## R Dependencies
The privLCM package uses R. It requires R (>= 4.0) and 
several R packages. The packages can be installed by starting 
R with the command 
```bash
R
```
and running
```R
install.package("plyr")
install.package("data.table")
install.package("rfast")
install.package("parallel")
install.package("./BayesLCM")
```

## Other code
The files `lib/mst.py` and `lib/privbayes.py` are from 
[https://github.com/ryan112358/private-pgm](https://github.com/ryan112358/private-pgm), 
with minor modifications.

# Running the Code

We use [Snakemake](https://snakemake.readthedocs.io/en/stable/)
to manage running our experiments. The command 
```bash
snakemake -j 6
```
runs all of our experiments using 6 cores in parallel. The 
number after `-j` sets the number of cores. Note that this 
will take several days on a single computer.

Figures for the toy data experiment are places in 
`latex/figures`, figures for the Adult and US Census experiments are 
placed in subdirectories. The figures 
will also be in the generated notebooks 
`processed_report-toy-data.py.ipynb`,
`processed_report-adult-reduced.py.ipynb`
and `processed_report-us-census.py.ipynb`.

The file `workflow/Snakemake` specifies the Adult data 
experiments that 
are run with the `repeats`, `epsilons` and `algorithms` 
variables. These can be edited to run a subset of the experiments,
or a smaller number of repeats. The toy data experiment is 
controlled in the same way by `workflow/rules/toy-data.smk`,
and the US Census experiment by `workflow/rules/us-census.smk`.

# US Census Data

The original data for the US Census experiment is fairly large
so we omitted it from the repository. It can be downloaded from 
the UCI repository
[https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/](https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/) (`USCensus1990.data.txt`).

The subset of the dataset used in the experiment is included,
so downloading the original dataset is not necessary to run 
the experiments.

# Navigating the Code

The implementation of NAPSU-MQ is in the `lib` directory.
The code refers to NAPSU-MQ with "maximum entropy", shortened to 
"max ent".

The `workflow` directory contains scripts that run the experiments
using Snakemake.

`adult-test.ipynb` and `max-ent-test.ipynb` are example notebooks 
for adult data and toy data experiments, respectively.