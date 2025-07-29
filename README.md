# Demokratieförderquote

_This is the output of a data project within the context of the [Civic Data Lab](https://civic-data.de). The Civic Data Lab team supported the AWO team with technical implementation of algorithms._

Links:
- [repository for crawling Förderdatenbank](https://github.com/CorrelAid/cdl_funding_crawler)
- [Förderdatenbank - Bund, Länder und EU](https://foerderdatenbank.de)

This repo contains
- a final static version of retrieved democracy funding programs: see X
- code for the development and evaluation of a classifier to automatically detect democracy funding programs 

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

The last codebook development and coding iteration was based on results of applying methods from the [cleanlab](https://github.com/cleanlab/cleanlab) libary. The AWO experts were asked to correct potential label issues and use unclear cases to refine the instructions in the codebook. With this final version of the codebook, AWO tasked to coders to label all already labeled data. This allows to determine inter-coder reliability and human labeling performance (see `notebooks/compute_icr.pdf` for details). 
ed on (McHugh (2012))[https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/] we use Percent Agreement as the main metric for inter-coder reliability, which was 89.3% and reflects strong reliability. Computing category-specific agreement showed that negatives have a much higher agreement (93.1%) than positives (75.2%). Setting the dataset as the ground truth that the two AWO experts cooperatively labeled and corrected in iterations, the two independent coders achieved an average accuracy of 88% and an average F1 score of 0.76. Taking a look at cases where both coders disagreed with the previous labeling, we hypothesize that errors were systematic and not random. Therefore, while the cooperatively created and iteratively corrected labeled data likely contains errors as well (making the problem of limited data even more challenging), is is used as the ground truth for final classifier development.

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