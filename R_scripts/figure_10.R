#create and bind all data except for early fusion, which will be added later
library(tidyverse)
log_reg_scores <- read_csv(
    "pipelines/results/ml_algorithms/log_reg/log_reg_test_scores.csv"
)
log_reg_scores <- log_reg_scores %>%
    mutate("algorithm" = "Logistic_regression") %>%
    select(c("drug", "features", "accuracy", "algorithm"))

svm_paper_scores <- read_csv(
    "pipelines/results/ml_algorithms/svm_paper/svm_paper_test_scores.csv"
)
svm_paper_scores <- svm_paper_scores %>%
    mutate("algorithm" = "SVC_paper") %>%
    select(c("drug", "features", "accuracy", "algorithm"))

intermediate_fusion_scores <- read_csv(
    "pipelines/results/neural_networks/intermediate_fusion_scores.csv"
)
intermediate_fusion_scores <- intermediate_fusion_scores %>%
    mutate("algorithm" = "intermediate_fusion") %>%
    select(c("drug", "features", "accuracy", "algorithm"))

late_fusion_scores <- read_csv(
    "pipelines/results/neural_networks/late_fusion_scores.csv"
)
late_fusion_scores <- late_fusion_scores %>%
    mutate("algorithm" = "late_fusion") %>%
    select(c("drug", "features", "accuracy", "algorithm"))

intermediate_fusion_scores_rf <- read_csv(
    "pipelines/results/neural_networks/intermediate_fusion_rf_scores.csv"
)
intermediate_fusion_scores_rf <- intermediate_fusion_scores_rf %>%
    mutate("algorithm" = "intermediate_fusion_rf") %>%
    select(c("drug", "features", "accuracy", "algorithm"))

late_fusion_scores_rf <- read_csv(
    "pipelines/results/neural_networks/late_fusion_rf_scores.csv"
)
late_fusion_scores_rf <- late_fusion_scores_rf %>%
    mutate("algorithm" = "late_fusion_rf") %>%
    select(c("drug", "features", "accuracy", "algorithm"))


all_scores = bind_rows(log_reg_scores, svm_paper_scores, intermediate_fusion_scores,
                        intermediate_fusion_scores_rf, late_fusion_scores, late_fusion_scores_rf)


#create early fusion datasets for the four drugs, keeping the score obtained using all features, and the best score in general
early_fusion_scores <- read_csv(
    "pipelines/results/neural_networks/early_fusion_scores.csv"
)
early_fusion_scores_rf <- read_csv(
    "pipelines/results/neural_networks/early_fusion_rf_scores.csv"
)


early_fusion_scores_cef <- early_fusion_scores %>%
    filter(drug == "Cef") %>%
    filter(accuracy == max(accuracy) | features == "genexp+gpa+snps") %>%
    mutate("algorithm" = ifelse(features == "genexp+gpa+snps", "early_fusion", "early_fusion_best_score")) %>%
    select(c("drug", "features", "accuracy", "algorithm"))
if(ncol(early_fusion_scores_cef) == 1){ #create a second row identical to the first one if the best score is achieved using all three features
    early_fusion_scores_cef = bind_rows(early_fusion_scores_cef, early_fusion_scores_cef)
    early_fusion_scores_cef$algorithm[2] = "early_fusion_best_score"
}

early_fusion_scores_rf_cef <- early_fusion_scores_rf %>%
    filter(drug == "Cef") %>%
    filter(accuracy == max(accuracy) | features == "genexp+gpa+snps") %>%
    mutate("algorithm" = ifelse(features == "genexp+gpa+snps", "early_fusion_rf", "early_fusion_rf_best_score")) %>%
    select(c("drug", "features", "accuracy", "algorithm"))
if(nrow(early_fusion_scores_rf_cef) == 1){ #create a second row identical to the first one if the best score is achieved using all three features
    early_fusion_scores_rf_cef = bind_rows(early_fusion_scores_rf_cef, early_fusion_scores_rf_cef)
    early_fusion_scores_rf_cef$algorithm[2] = "early_fusion_rf_best_score"
}


early_fusion_scores_cip <- early_fusion_scores %>%
    filter(drug == "Cip") %>%
    filter(accuracy == max(accuracy) | features == "genexp+gpa+snps") %>%
    mutate("algorithm" = ifelse(features == "genexp+gpa+snps", "early_fusion", "early_fusion_best_score")) %>%
    select(c("drug", "features", "accuracy", "algorithm"))
if(ncol(early_fusion_scores_cip) == 1){ #create a second row identical to the first one if the best score is achieved using all three features
    early_fusion_scores_cip = bind_rows(early_fusion_scores_cip, early_fusion_scores_cip)
    early_fusion_scores_cip$algorithm[2] = "early_fusion_best_score"
}

early_fusion_scores_rf_cip <- early_fusion_scores_rf %>%
    filter(drug == "Cip") %>%
    filter(accuracy == max(accuracy) | features == "genexp+gpa+snps") %>%
    mutate("algorithm" = ifelse(features == "genexp+gpa+snps", "early_fusion_rf", "early_fusion_rf_best_score")) %>%
    select(c("drug", "features", "accuracy", "algorithm"))
if(nrow(early_fusion_scores_rf_cip) == 1){ #create a second row identical to the first one if the best score is achieved using all three features
    early_fusion_scores_rf_cip = bind_rows(early_fusion_scores_rf_cip, early_fusion_scores_rf_cip)
    early_fusion_scores_rf_cip$algorithm[2] = "early_fusion_rf_best_score"
}


early_fusion_scores_mer <- early_fusion_scores %>%
    filter(drug == "Mer") %>%
    filter(accuracy == max(accuracy) | features == "genexp+gpa+snps") %>%
    mutate("algorithm" = ifelse(features == "genexp+gpa+snps", "early_fusion", "early_fusion_best_score")) %>%
    select(c("drug", "features", "accuracy", "algorithm"))
if(ncol(early_fusion_scores_mer) == 1){ #create a second row identical to the first one if the best score is achieved using all three features
    early_fusion_scores_mer = bind_rows(early_fusion_scores_mer, early_fusion_scores_mer)
    early_fusion_scores_mer$algorithm[2] = "early_fusion_best_score"
}

early_fusion_scores_rf_mer <- early_fusion_scores_rf %>%
    filter(drug == "Mer") %>%
    filter(accuracy == max(accuracy) | features == "genexp+gpa+snps") %>%
    mutate("algorithm" = ifelse(features == "genexp+gpa+snps", "early_fusion_rf", "early_fusion_rf_best_score")) %>%
    select(c("drug", "features", "accuracy", "algorithm"))
if(nrow(early_fusion_scores_rf_mer) == 1){ #create a second row identical to the first one if the best score is achieved using all three features
    early_fusion_scores_rf_mer = bind_rows(early_fusion_scores_rf_mer, early_fusion_scores_rf_mer)
    early_fusion_scores_rf_mer$algorithm[2] = "early_fusion_rf_best_score"
}


early_fusion_scores_tob <- early_fusion_scores %>%
    filter(drug == "Tob") %>%
    filter(accuracy == max(accuracy) | features == "genexp+gpa+snps") %>%
    mutate("algorithm" = ifelse(features == "genexp+gpa+snps", "early_fusion", "early_fusion_best_score")) %>%
    select(c("drug", "features", "accuracy", "algorithm"))
if(ncol(early_fusion_scores_tob) == 1){ #create a second row identical to the first one if the best score is achieved using all three features
    early_fusion_scores_tob = bind_rows(early_fusion_scores_tob, early_fusion_scores_tob)
    early_fusion_scores_tob$algorithm[2] = "early_fusion_best_score"
}

early_fusion_scores_rf_tob <- early_fusion_scores_rf %>%
    filter(drug == "Tob") %>%
    filter(accuracy == max(accuracy) | features == "genexp+gpa+snps") %>%
    mutate("algorithm" = ifelse(features == "genexp+gpa+snps", "early_fusion_rf", "early_fusion_rf_best_score")) %>%
    select(c("drug", "features", "accuracy", "algorithm"))
if(nrow(early_fusion_scores_rf_tob) == 1){ #create a second row identical to the first one if the best score is achieved using all three features
    early_fusion_scores_rf_tob = bind_rows(early_fusion_scores_rf_tob, early_fusion_scores_rf_tob)
    early_fusion_scores_rf_tob$algorithm[2] = "early_fusion_rf_best_score"
}


#merge everything in all_scores dataset
all_scores <- bind_rows(all_scores, early_fusion_scores_cef, early_fusion_scores_cip, early_fusion_scores_mer, early_fusion_scores_tob,
                        early_fusion_scores_rf_cef, early_fusion_scores_rf_cip, early_fusion_scores_rf_mer, early_fusion_scores_rf_tob)


#create a plot of scores for each drug
cef_scores <- all_scores %>%
    filter(drug == 'Cef')

cef_plot <- ggplot(data = cef_scores, aes(x = algorithm, y = accuracy, fill = features)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("genexp" = "#66c2a5", "gpa" = "#fc8d62", "snps" = "#8da0cb", 
                        "genexp+gpa" = "#e78ac3", "genexp+snps" = "#a6d854", "gpa+snps"="#ffd92f", "genexp+gpa+snps" = "#e5c494")) +
    labs(title = "Classification scores for Ceftazidim", x = "Algorithm/Neural network", y = "Accuracy score", fill = "Combination of features") +
    geom_text(aes(label = round(accuracy, digits = 2), vjust = 2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("plots/figure_10/cef_scores.png", plot = cef_plot, width = 6.67, height = 6.67)

cip_scores <- all_scores %>%
    filter(drug == 'Cip')

cip_plot <- ggplot(data = cip_scores, aes(x = algorithm, y = accuracy, fill = features)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("genexp" = "#66c2a5", "gpa" = "#fc8d62", "snps" = "#8da0cb", 
                        "genexp+gpa" = "#e78ac3", "genexp+snps" = "#a6d854", "gpa+snps"="#ffd92f", "genexp+gpa+snps" = "#e5c494")) +
    labs(title = "Classification scores for Ciprofloxacin", x = "Algorithm/Neural network", y = "Accuracy score", fill = "Combination of features") +
    geom_text(aes(label = round(accuracy, digits = 2), vjust = 2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("plots/figure_10/cip_scores.png", plot = cip_plot, width = 6.67, height = 6.67)

mer_scores <- all_scores %>%
    filter(drug == 'Mer')

mer_plot <- ggplot(data = mer_scores, aes(x = algorithm, y = accuracy, fill = features)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("genexp" = "#66c2a5", "gpa" = "#fc8d62", "snps" = "#8da0cb", 
                        "genexp+gpa" = "#e78ac3", "genexp+snps" = "#a6d854", "gpa+snps"="#ffd92f", "genexp+gpa+snps" = "#e5c494")) +
    labs(title = "Classification scores for Meropenem", x = "Algorithm/Neural network", y = "Accuracy score", fill = "Combination of features") +
    geom_text(aes(label = round(accuracy, digits = 2), vjust = 2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("plots/figure_10/mer_scores.png", plot = mer_plot, width = 6.67, height = 6.67)

tob_scores <- all_scores %>%
    filter(drug == 'Tob')

tob_plot <- ggplot(data = tob_scores, aes(x = algorithm, y = accuracy, fill = features)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("genexp" = "#66c2a5", "gpa" = "#fc8d62", "snps" = "#8da0cb", 
                        "genexp+gpa" = "#e78ac3", "genexp+snps" = "#a6d854", "gpa+snps"="#ffd92f", "genexp+gpa+snps" = "#e5c494")) +
    labs(title = "Classification scores for Tobramycin", x = "Algorithm/Neural network", y = "Accuracy score", fill = "Combination of features") +
    geom_text(aes(label = round(accuracy, digits = 2), vjust = 2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("plots/figure_10/tob_scores.png", plot = tob_plot, width = 6.67, height = 6.67)