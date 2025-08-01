# Demokratieförderquote

_This is the output of a data project within the context of the [Civic Data Lab](https://civic-data.de). The Civic Data Lab team supported the AWO team with technical implementation of algorithms._

Links:
- [repository for crawling Förderdatenbank](https://github.com/CorrelAid/cdl_funding_crawler)
- [Förderdatenbank - Bund, Länder und EU](https://foerderdatenbank.de)

This repo contains
- a final static version of democracy funding programs on federal level. See `data/labeled/final/2025_07_21_Tabelle_Programme_zur_Publ.xlsx`
- code for the development and evaluation of a classifier to automatically detect democracy funding programs 

## License
### Code

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

see `LICENSE-CODE`

### Data 

We refer to the [imprint of foerderdatenbank.de](https://www.foerderdatenbank.de/FDB/DE/Meta/Impressum/impressum.html) of the Federal Ministry for Economic Affairs and Climate Action which indicates [CC BY-ND 3.0 DE](https://creativecommons.org/licenses/by-nd/3.0/de/deed.de) as the license for all texts of the website. The dataset provided in this repository transfers information on each funding program into a machine-readable format.


see `LICENSE-DATA`

## Method and Evaluation Results

The task was to develop a binary text classifier with limited and highly imbalanced data (with many more negatives than positives). Moreover, the concept of what constitutes a positive case is rather complex.

The Codebook (see `codebook`) defines what is understood as a democracy funding program within this project. It was developed iteratively through strategic data selection in collaboration with two experts from AWO. Given the low number of positives, it was crucial to first retrieve as many positives as possible. In an initial iteration, we selected a couple of dozen funding programs based on the results of a semantic search using an early version of the codebook. This was extended by results from an inverse semantic search to retrieve potential hard negatives. Subsequent coding iterations were based on predicted positives from early classifier results and random sampling of predicted negatives. 

The final codebook development and coding iteration incorporated methods from the [cleanlab](https://github.com/cleanlab/cleanlab) library. AWO experts were asked to review potential label issues identified by cleanlab and to refine the codebook instructions using unclear cases. See Using this final codebook, AWO tasked coders to label all previously labeled data, enabling us to measure inter-coder reliability and human labeling performance (see `notebooks/compute_icr.pdf` for details).

Based on (McHugh (2012))[https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/], we used Percent Agreement as the main metric for inter-coder reliability, which was 89.3%, reflecting strong agreement. Category-specific agreement showed that negatives had much higher agreement (93.1%) than positives (75.2%). Considering the dataset as ground truth—labeled and corrected iteratively by the two AWO experts—two independent coders achieved an average accuracy of 88% and an average F1 score of 0.76. Analysis of cases where both coders disagreed with the previous labeling suggests systematic, rather than random, errors. Therefore, the cooperatively created and iteratively refined labeled dataset was used as ground truth for final classifier development.

We evaluated the following methods:
- Rule-Based (e.g., classifying a program if it contains the word “democracy”)
- TF–IDF as input to a Multilayer Perceptron (MLP)
- In-Context Learning (ICL)
- FastFit
- Fine-tuning a further pretrained BERT model

To address class imbalance and the small dataset size, we tested several techniques, most notably oversampling with SMOTE and data extension through paraphrasing with large language models (LLMs).

Methods were mostly evaluated using a fixed 10-split shuffle split. The rule-based method was evaluated on the entire dataset; ICL was evaluated on 80% of the data since only 20% is needed for the MIPRO algorithm (as recommended by [dspy](https://dspy.ai/learn/optimization/overview/)). Hyperparameter optimization was only conducted for the TF-IDF + MLP model due to resource constraints with LLM-based methods. For shuffle split evaluations, we report the mean metric values.

For ICL, applying MIPRO  with a medium study size worsened performance compared to zero-shot ICL. FastFit and fine-tuning of a further pretrained BERT performed worse than other methods. Among all approaches, the best performance was achieved by the TF-IDF + MLP model.

| Experiment            | Model/Variant                      | F1     | Accuracy | Precision | Recall  | F1 Std | Acc Std | Prec Std | Rec Std |
|-----------------------|----------------------------------|--------|----------|-----------|---------|--------|---------|----------|---------|
| TF-IDF + MLP          | Optuna tuned                     | 0.770  | 0.875    | 0.826     | 0.726   | 0.045  | 0.021   | 0.038    | 0.076   |
| Fine-tuned BERT       | –                                | 0.760  | 0.814    | 0.708     | 0.863   | 0.120  | 0.187   | 0.163    | 0.092   |
| ICL (Zero-Shot)       | moonshotai/kimi-k2               | 0.757  | 0.875    | 0.875     | 0.667   | –      | –       | –        | –       |
| ICL + MIPRO           | moonshotai/kimi-k2               | 0.733  | 0.852    | 0.772     | 0.698   | –      | –       | –        | –       |
| FastFit               | –                                | 0.725  | 0.844    | 0.751     | 0.711   | 0.077  | 0.045   | 0.097    | 0.103   |
| Rule-Based            | description_text_only             | 0.667  | 0.748    | 0.538     | 0.876   | –      | –       | –        | –       |


The best model (TF-IDF + MLP) already performs better than humans on the labeling task when setting expert labels as ground truth. Still, performance is not sufficient for the use case of the classifier, which is using classification results for further analysis directly. Accepting that manual work is necessary, optimizing the model for recall is an option, since recall is more important than precision because missing positives would distort the analysis in a more problematic direction. It is feasible to manually verify predicted positives but not negatives. We use the strategy to optimize recall until 0.95 and then prioritize precision. The results is a recall of 0.95 and a precision of 0.57. This model should be used as assistance and its results manually verified.


## Setup

### Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Dev Setup

1. Clone this repository
2. `uv sync`
3. `uv run pre-commit install`
