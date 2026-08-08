library(tidyverse)


sweep_svc_sigmoid <- read_csv(
    "pipelines/results/ml_algorithms/sweep/svc_sigmoid_sweep_gamma_scores.csv"
)
sweep_svc_rbf <- read_csv(
    "pipelines/results/ml_algorithms/sweep/svc_rbf_sweep_gamma_scores.csv"
)

plot <- ggplot() +
    scale_color_manual(values = c("SVC sigmoidal" = "#b07aa1", "SVC Gaussian"="#edc948")) +
    geom_line(data = sweep_svc_sigmoid, aes(x = log10(Gamma), y = Test_score, color = "SVC sigmoidal"), linewidth = 1) +
    geom_point(data = sweep_svc_sigmoid, aes(x = log10(Gamma), y = Test_score, color = "SVC sigmoidal"), size = 3) +
    geom_line(data = sweep_svc_rbf, aes(x = log10(Gamma), y = Test_score, color = "SVC Gaussian"), linewidth = 1) +
    geom_point(data = sweep_svc_rbf, aes(x = log10(Gamma), y = Test_score, color = "SVC Gaussian"), size = 3) +
    labs(title = "Test scores all features Ceftazidim", x = "Gamma (log10)", y = "Accuracy score", color = "Algorithm") +
    theme_bw()

ggsave("plots/figure_6.png", plot = plot, width = 6.67, height = 6.67)