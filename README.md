# nlp_assignment_3
Files and code for NLP assignment 3. Pipeline in Google Colab notebook due to lack of power on student laptop. Data analysis in R.

## Python dependencies
The code requires the following Python libraries:

| Library | Purpose |
|----------|----------|
| pandas   | Creating and handling dataset     |
| transformers   | Handling LLMs     |
| torch   | Disabling gradient calculation     |
| google.colab   | Saving output files     |
| huggingface_hub   | Logging in to huggingface to use Llama     |
| requests   | Reading in conllu (Treebank) file     |
| conllu   | Handling conllu files     |
| re   | Using regular expressions     |

## R dependencies
The code requires the following R libraries:

| Library | Purpose |
|----------|----------|
| tidyverse   | Ensuring tidy data     |
| lme4   | Fitting generalised mixed effects models     |
| ggplot2   | Visualising models as plots     |
| marginaleffects   | Interpreting models     |
| flextable   | Producing tables     |

## Other resources needed
### Treebank files
https://raw.githubusercontent.com/bethancunningham/tfm/main/treebank_train.conllu
https://raw.githubusercontent.com/bethancunningham/tfm/main/treebank_dev.conllu
https://raw.githubusercontent.com/bethancunningham/tfm/main/treebank_test.conllu

### Final dataset for pipeline
https://raw.githubusercontent.com/bethancunningham/nlp_2026/main/dataset_assignment3.csv
