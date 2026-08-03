library(tidyverse)

#load data (except for Gaussian and sigmoidal SVC) and create a column with the machine learning algorithm used 
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


#unify all datasets into a single one, and keep only Meropenem
all_scores <- bind_rows(knn_scores, lda_scores, svc_linear_scores, svc_poly_scores, log_reg_scores, svm_paper_scores)
all_scores <- all_scores %>%
    filter(drug == 'Mer')


#plot recall of susceptible samples for Meropenem
p <- ggplot(data = all_scores, aes(x = algorithm, y = recall_s, fill = features)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    scale_fill_manual(values = c("genexp" = "#ff0000", "gpa" = "#ffA500", "snps" = "#ffff00", 
                        "genexp+gpa" = "#32cd32", "genexp+snps" = "#0000ff", "gpa+snps"="#63e6f8", "genexp+gpa+snps" = "#ee82ee")) +
    labs(title = "Recall susceptibles Meropenem", x = "Machine learning algorithm", y = "Accuracy score", fill = "Best combination of features") +
    geom_text(aes(label = round(recall_s, digits = 2), hjust = 1.2)) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    theme_bw()
ggsave("plots/figure_5.png", plot = p, width = 6.67, height = 6.67)