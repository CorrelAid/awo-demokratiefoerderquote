# Demokratieförderquote\*

This repo contains
- a final static version of retrieved democracy funding programs: see [data/2025_07_21_Tabelle_Programme_zur_Publ.xlsx](https://github.com/CorrelAid/awo-demokratiefoerderquote/blob/main/data/2025_07_21_Tabelle_Programme_zur_Publ.xlsx) 
- code for the development and evaluation of a classifier to automatically detect democracy funding programs 

This repository is part of a project of the AWO Bundesverband e.V. with technical support provided by the [Civic Data Lab](https://civic-data.de) team.

Related links:
- [Förderdatenbank Crawler](https://github.com/CorrelAid/cdl_funding_crawler)
- [Förderdatenbank - Bund, Länder und EU](https://foerderdatenbank.de)

\*Democracy funding quota

## License
### Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

see `LICENSE-CODE`

### Data 

We refer to the [imprint of foerderdatenbank.de](https://www.foerderdatenbank.de/FDB/DE/Meta/Impressum/impressum.html) of the Federal Ministry for Economic Affairs and Climate Action which indicates [CC BY-ND 3.0 DE](https://creativecommons.org/licenses/by-nd/3.0/de/deed.de) as the license for all texts of the website. The dataset provided in this repository transfers information on each funding program into a machine-readable format.


see `LICENSE-DATA`

## Method and Evaluation Results

The task was to develop a binary text classifier with limited and highly imbalanced data (much more negatives than positives). Furthermore, the concept of what is a positive is rather complex.
 
The Codebook (see `codebook`) specifies what is understood as a democracy funding program in the context of this project. It was developed in iterations based on strategic data selection with two experts from AWO. In a first iteration, we selected a couple dozen funding programs based on the results of semantic search with an early version of the codebook. This was extended with results of an inverse semantic search to retreive potential hard negatives. Later coding iterations were based on predicted positives from the results of early classifiers and random sampling of predicted negatives. The file `data/labeled/19_06/legacy.json` is the result of this process. 

The last codebook development and coding iteration was based on results of applying methods from the [cleanlab](https://github.com/cleanlab/cleanlab) libary. The AWO experts were asked to correct potential label issues and use unclear cases to refine the instructions in the codebook. With this final version of the codebook, AWO tasked two coders to label all already labeled data. This allows to determine inter-coder reliability and human labeling performance (see `notebooks/compute_icr.pdf` for details). 
ed on (McHugh (2012))[https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/] we use Percent Agreement as the main metric for inter-coder reliability, which was 89.3% and reflects strong reliability. 

## Setup

### Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Dev Setup

1. Clone this repository
2. `uv sync`
3. `uv run pre-commit install`


### Last steps
1. Implement One data augmentation strategy -> nlpaug backtranslation
2. Add oversample parameter to optimization (with smote and data augmentation), use especially for recall
3. test fastfit
4. retrain bert, 
    - if significantly better than tfidf, adjust README add license and publish
