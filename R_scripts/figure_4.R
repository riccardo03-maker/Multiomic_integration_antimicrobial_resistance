library(tidyverse)

#load test scores of classical machine learning algorithms and create a column with the machine learning algorithm used
knn_scores <- read_csv(
    "pipelines/results/ml_algorithms/knn/knn_test_scores.csv"
)
knn_scores <- knn_scores %>%
    mutate("algorithm" = "KNN")

lda_scores <- read_csv(
    "pipelines/results/ml_algorithms/lda/lda_test_scores.csv"
)
lda_scores <- lda_scores %>%
    mutate("algorithm" = "LDA")

svc_linear_scores <- read_csv(
    "pipelines/results/ml_algorithms/svc_linear/svc_linear_test_scores.csv"
)
svc_linear_scores <- svc_linear_scores %>%
    mutate("algorithm" = "SVC_linear")

svc_poly_scores <- read_csv(
    "pipelines/results/ml_algorithms/svc_poly/svc_poly_test_scores.csv"
)
svc_poly_scores <- svc_poly_scores %>%
    mutate("algorithm" = "SVC_poly")

svc_rbf_scores <- read_csv(
    "pipelines/results/ml_algorithms/svc_rbf/svc_rbf_test_scores.csv"
)
svc_rbf_scores <- svc_rbf_scores %>%
    mutate("algorithm" = "SVC_gaussian")

svc_sigmoid_scores <- read_csv(
    "pipelines/results/ml_algorithms/svc_sigmoid/svc_sigmoid_test_scores.csv"
)
svc_sigmoid_scores <- svc_sigmoid_scores %>%
    mutate("algorithm" = "SVC_sigmoidal")

log_reg_scores <- read_csv(
    "pipelines/results/ml_algorithms/log_reg/log_reg_test_scores.csv"
)
log_reg_scores <- log_reg_scores %>%
    mutate("algorithm" = "Logistic_regression")

svm_paper_scores <- read_csv(
    "pipelines/results/ml_algorithms/svm_paper/svm_paper_test_scores.csv"
)
svm_paper_scores <- svm_paper_scores %>%
    mutate("algorithm" = "SVC_paper")


#unify all datasets into a single one
all_scores <- bind_rows(knn_scores, lda_scores, svc_linear_scores, svc_poly_scores, svc_rbf_scores, 
                        svc_sigmoid_scores, log_reg_scores, svm_paper_scores)


#create a plot of scores for each drug
cef_scores <- all_scores %>%
    filter(drug == 'Cef')

cef_plot <- ggplot(data = cef_scores, aes(x = algorithm, y = accuracy, fill = features)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("genexp" = "#ff0000", "gpa" = "#ffA500", "snps" = "#ffff00", 
                        "genexp+gpa" = "#32cd32", "genexp+snps" = "#0000ff", "gpa+snps"="#63e6f8", "genexp+gpa+snps" = "#ee82ee")) +
    labs(title = "Classification scores for Ceftazidim", x = "Machine learning algorithm", y = "Accuracy score", fill = "Best combination of features") +
    geom_text(aes(label = round(accuracy, digits = 2), vjust = 2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("plots/figure_4/cef_scores.png", plot = cef_plot, width = 6.67, height = 6.67)

cip_scores <- all_scores %>%
   filter(drug == 'Cip')

cip_plot <- ggplot(data = cip_scores, aes(x = algorithm, y = accuracy, fill = features)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("genexp" = "#ff0000", "gpa" = "#ffA500", "snps" = "#ffff00", 
                        "genexp+gpa" = "#32cd32", "genexp+snps" = "#0000ff", "gpa+snps"="#63e6f8", "genexp+gpa+snps" = "#ee82ee")) +
    labs(title = "Classification scores for Ciprofloxacin", x = "Machine learning algorithm", y = "Accuracy score", fill = "Best combination of features") +
    geom_text(aes(label = round(accuracy, digits = 2), vjust = 2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("plots/figure_4/cip_scores.png", plot = cip_plot, width = 6.67, height = 6.67)

mer_scores <- all_scores %>%
   filter(drug == 'Mer')

mer_plot <- ggplot(data = mer_scores, aes(x = algorithm, y = accuracy, fill = features)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("genexp" = "#ff0000", "gpa" = "#ffA500", "snps" = "#ffff00", 
                        "genexp+gpa" = "#32cd32", "genexp+snps" = "#0000ff", "gpa+snps"="#63e6f8", "genexp+gpa+snps" = "#ee82ee")) +
    labs(title = "Classification scores for Meropenem", x = "Machine learning algorithm", y = "Accuracy score", fill = "Best combination of features") +
    geom_text(aes(label = round(accuracy, digits = 2), vjust = 2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("plots/figure_4/mer_scores.png", plot = mer_plot, width = 6.67, height = 6.67)

tob_scores <- all_scores %>%
   filter(drug == 'Tob')

tob_plot <- ggplot(data = tob_scores, aes(x = algorithm, y = accuracy, fill = features)) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = c("genexp" = "#ff0000", "gpa" = "#ffA500", "snps" = "#ffff00", 
                        "genexp+gpa" = "#32cd32", "genexp+snps" = "#0000ff", "gpa+snps"="#63e6f8", "genexp+gpa+snps" = "#ee82ee")) +
    labs(title = "Classification scores for Tobramycin", x = "Machine learning algorithm", y = "Accuracy score", fill = "Best combination of features") +
    geom_text(aes(label = round(accuracy, digits = 2), vjust = 2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave("plots/figure_4/tob_scores.png", plot = tob_plot, width = 6.67, height = 6.67) 