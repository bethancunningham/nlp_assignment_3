library(tidyverse)
library(lme4)
library(ggplot2)
library(marginaleffects)
library(flextable)

# Reading in results csvs

url_goldfish <- "https://raw.githubusercontent.com/bethancunningham/nlp_2026/main/goldfish_results.csv"

url_britllm <- "https://raw.githubusercontent.com/bethancunningham/nlp_2026/main/britllm_results.csv"

url_llama_1 <- "https://raw.githubusercontent.com/bethancunningham/nlp_2026/main/llama_first_250.csv"

url_llama_2 <- "https://raw.githubusercontent.com/bethancunningham/nlp_2026/main/llama_final_50.csv"

df_goldfish <- read.csv(url_goldfish)
df_britllm <- read.csv(url_britllm)
df_llama_1 <- read.csv(url_llama_1)
df_llama_2 <- read.csv(url_llama_2)


# Making 1 df from the above 4, adding 'correct' column (1 if correct NLL lower than incorrect NLL, otherwise 0), changing model names and converting model and sentence_id to factors

df <- bind_rows(df_goldfish, df_britllm, df_llama_1, df_llama_2) %>%
  mutate(
    correct = ifelse(p_correct < p_incorrect, 1, 0),
    model = recode(model,
                   "britllm/britllm-3b-v0.1" = "BritLLM",
                   "goldfish-models/cym_latn_10mb" = "Goldfish",
                   "meta-llama/Llama-3.1-8B" = "Llama"
    ),
    model = factor(model),
    sentence_id = factor(sentence_id)
  )

# Changing name of 'model' column due to confusion with regression models later

names(df)[names(df) == "model"] <- "LM"

# Accuracy by model

table1 <- df |>
  group_by(LM) |>
  summarise(accuracy = mean(correct) * 100)

# Accuracy by trigger type

table2 <- df |>
  group_by(trigger_type) |>
  summarise(accuracy = mean(correct) * 100)

# Accuracy by model and trigger type

table3 <- df |>
  group_by(LM, trigger_type) |>
  summarise(accuracy = mean(correct) * 100)

flextable(table1)
flextable(table2)
flextable(table3)

# Which sentences did 2 or all LMs get wrong?

df |> 
  group_by(sentence_id) |> 
  filter(sum(correct) <= 1) |> 
  View()

# Table of number of sentences where 2 or all LMs got it wrong in terms of trigger type

df |> 
  group_by(sentence_id) |> 
  filter(sum(correct) <= 1) |> 
  distinct(sentence_id, .keep_all = TRUE) |>
  group_by(trigger_type) |>
  summarise(n = n())

# Fitting generalised linear model to estimate effect of LM on probability of correct answer, then visualising in probability space

model_1 <- glmer(correct ~ LM + (1 | sentence_id), data = df, family = binomial)

plot_1 <- plot_predictions(model_1, condition = "LM", type = "response")

# Changing labels

plot_1 <- plot_1 +
  labs(
    y = "Probability of correct answer",
    x = "Model",
    )

print(plot_1)

# Getting p-values for pairwise differences

avg_comparisons(model_1, variables = list(LM = "pairwise"))


# Same again but effect of mutation trigger type

model_2 <- glmer(
  correct ~ trigger_type + (1 | sentence_id), data = df,  family = binomial)

plot_2 <- plot_predictions(
  model_2,
  condition = "trigger_type",
)

# Getting p-values for pairwise differences

avg_comparisons(model_2, variables = list(trigger_type = "pairwise"))

# Changing labels

plot_2 <- plot_2 +
  labs(
    y = "Probability of correct answer",
    x = "Mutation trigger type"
  ) 

print(plot_2)

# Same again but effects of trigger type and LM and interaction between them

model_3 <- glmer(
  correct ~ trigger_type * LM + (1 | sentence_id), data = df,  family = binomial)

plot_3 <- plot_predictions(
  model_3,
  condition = c("trigger_type", "LM"),
)

# Changing labels

plot_3 <- plot_3 +
  labs(
    y = "Probability of correct answer",
    x = "Mutation trigger type",
    color = "Model"
  )

print(plot_3)
