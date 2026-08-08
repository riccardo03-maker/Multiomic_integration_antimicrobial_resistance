library(tidyverse)

#load data
sweep_log_reg <- read_csv(
    "pipelines/results/ml_algorithms/sweep/log_reg_sweep_C_scores.csv"
)
sweep_svm_paper <- read_csv(
    "pipelines/results/ml_algorithms/sweep/svm_paper_sweep_C_scores.csv"
)
sweep_svc_linear <- read_csv(
    "pipelines/results/ml_algorithms/sweep/svc_linear_sweep_C_scores.csv"
)
sweep_svc_poly <- read_csv(
    "pipelines/results/ml_algorithms/sweep/svc_poly_sweep_C_scores.csv"
)
sweep_svc_sigmoid <- read_csv(
    "pipelines/results/ml_algorithms/sweep/svc_sigmoid_sweep_C_scores.csv"
)
sweep_svc_rbf <- read_csv(
    "pipelines/results/ml_algorithms/sweep/svc_rbf_sweep_C_scores.csv"
)

#plot cross validation scores
plot <- ggplot() +
    scale_color_manual(values = c("Logistic Regression" = "#4e79a7", "SVC paper" = "#f28e2b", "SVC linear" = "#59a14f", 
                        "SVC poly" = "#e15759", "SVC sigmoidal" = "#b07aa1", "SVC Gaussian"="#edc948")) +
    geom_line(data = sweep_log_reg, aes(x = log10(C), y = Cv_score, color = "Logistic Regression"), linewidth = 1, ) +
    geom_point(data = sweep_log_reg, aes(x = log10(C), y = Cv_score, color = "Logistic Regression"), size = 3) +
    geom_line(data = sweep_svm_paper, aes(x = log10(C), y = Cv_score, color = "SVC paper"), linewidth = 1) +
    geom_point(data = sweep_svm_paper, aes(x = log10(C), y = Cv_score, color = "SVC paper"), size = 3) +
    geom_line(data = sweep_svc_linear, aes(x = log10(C), y = Cv_score, color = "SVC linear"), linewidth = 1) +
    geom_point(data = sweep_svc_linear, aes(x = log10(C), y = Cv_score, color = "SVC linear"), size = 3) +
    geom_line(data = sweep_svc_poly, aes(x = log10(C), y = Cv_score, color = "SVC poly"), linewidth = 1) +
    geom_point(data = sweep_svc_poly, aes(x = log10(C), y = Cv_score, color = "SVC poly"), size = 3) +
    geom_line(data = sweep_svc_sigmoid, aes(x = log10(C), y = Cv_score, color = "SVC sigmoidal"), linewidth = 1) +
    geom_point(data = sweep_svc_sigmoid, aes(x = log10(C), y = Cv_score, color = "SVC sigmoidal"), size = 3) +
    geom_line(data = sweep_svc_rbf, aes(x = log10(C), y = Cv_score, color = "SVC Gaussian"), linewidth = 1) +
    geom_point(data = sweep_svc_rbf, aes(x = log10(C), y = Cv_score, color = "SVC Gaussian"), size = 3) +
    labs(title = "Cross validation scores all features Ceftazidim", x = "C (log10)", y = "F1 macro score", color = "Algorithm") +
    theme_bw()

ggsave("plots/figure_4/sweep_crossval_scores.png", plot = plot, width = 6.67, height = 6.67)

#plot scores on test set
plot <- ggplot() +
    scale_color_manual(values = c("Logistic Regression" = "#4e79a7", "SVC paper" = "#f28e2b", "SVC linear" = "#59a14f", 
                        "SVC poly" = "#e15759", "SVC sigmoidal" = "#b07aa1", "SVC Gaussian"="#edc948")) +
    geom_line(data = sweep_log_reg, aes(x = log10(C), y = Test_score, color = "Logistic Regression"), linewidth = 1) +
    geom_point(data = sweep_log_reg, aes(x = log10(C), y = Test_score, color = "Logistic Regression"), size = 3) +
    geom_line(data = sweep_svm_paper, aes(x = log10(C), y = Test_score, color = "SVC paper"), linewidth = 1) +
    geom_point(data = sweep_svm_paper, aes(x = log10(C), y = Test_score, color = "SVC paper"), size = 3) +
    geom_line(data = sweep_svc_linear, aes(x = log10(C), y = Test_score, color = "SVC linear"), linewidth = 1) +
    geom_point(data = sweep_svc_linear, aes(x = log10(C), y = Test_score, color = "SVC linear"), size = 3) +
    geom_line(data = sweep_svc_poly, aes(x = log10(C), y = Test_score, color = "SVC poly"), linewidth = 1) +
    geom_point(data = sweep_svc_poly, aes(x = log10(C), y = Test_score, color = "SVC poly"), size = 3) +
    geom_line(data = sweep_svc_sigmoid, aes(x = log10(C), y = Test_score, color = "SVC sigmoidal"), linewidth = 1) +
    geom_point(data = sweep_svc_sigmoid, aes(x = log10(C), y = Test_score, color = "SVC sigmoidal"), size = 3) +
    geom_line(data = sweep_svc_rbf, aes(x = log10(C), y = Test_score, color = "SVC Gaussian"), linewidth = 1) +
    geom_point(data = sweep_svc_rbf, aes(x = log10(C), y = Test_score, color = "SVC Gaussian"), size = 3) +
    labs(title = "Test scores all features Ceftazidim", x = "C (log10)", y = "Accuracy score", color = "Algorithm") +
    theme_bw()

ggsave("plots/figure_4/sweep_test_scores.png", plot = plot, width = 6.67, height = 6.67)