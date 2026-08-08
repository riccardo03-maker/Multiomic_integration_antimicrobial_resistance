library(tidyverse)


loss_evolution_cef <- read_csv(
    "pipelines/nn_trained_models/intermediate_fusion/loss_function/intermediate_fusion_genexp_gpa_snps_Cef.csv"
)
loss_evolution_cef <- loss_evolution_cef %>%
    filter(Epoch >= 2)
#start from the third epoch for plottability

loss_evolution_cip <- read_csv(
    "pipelines/nn_trained_models/intermediate_fusion/loss_function/intermediate_fusion_genexp_gpa_snps_Cip.csv"
)
loss_evolution_cip <- loss_evolution_cip %>%
    filter(Epoch >= 2)

loss_evolution_mer <- read_csv(
    "pipelines/nn_trained_models/intermediate_fusion/loss_function/intermediate_fusion_genexp_gpa_snps_Mer.csv"
)
loss_evolution_mer <- loss_evolution_mer %>%
    filter(Epoch >= 2)

loss_evolution_tob <- read_csv(
    "pipelines/nn_trained_models/intermediate_fusion/loss_function/intermediate_fusion_genexp_gpa_snps_Tob.csv"
)
loss_evolution_tob <- loss_evolution_tob %>%
    filter(Epoch >= 2)


plot <- ggplot() +
    geom_line(data = loss_evolution_cef, aes(x = Epoch, y = Loss, color = "Ceftazidim"), linewidth = 1) +
    geom_point(data = loss_evolution_cef, aes(x = Epoch, y = Loss, color = "Ceftazidim"), size = 2) +
    geom_line(data = loss_evolution_cip, aes(x = Epoch, y = Loss, color = "Ciprofloxacin"), linewidth = 1) +
    geom_point(data = loss_evolution_cip, aes(x = Epoch, y = Loss, color = "Ciprofloxacin"), size = 2) +
    geom_line(data = loss_evolution_mer, aes(x = Epoch, y = Loss, color = "Meropenem"), linewidth = 1) +
    geom_point(data = loss_evolution_mer, aes(x = Epoch, y = Loss, color = "Meropenem"), size = 2) +
    geom_line(data = loss_evolution_tob, aes(x = Epoch, y = Loss, color = "Tobramycin"), linewidth = 1) +
    geom_point(data = loss_evolution_tob, aes(x = Epoch, y = Loss, color = "Tobramycin"), size = 2) +
    labs(title = "Loss evolution over epochs intermediate fusion", x = "Epoch", y = "Loss", color = "Drug") +
    theme_bw()

ggsave("plots/figure_9/intermediate_fusion.png", plot = plot, width = 6.67, height = 6.67)


#with relevant features selection
loss_evolution_cef <- read_csv(
    "pipelines/nn_trained_models/intermediate_fusion_rf/loss_function/intermediate_fusion_rf_genexp_gpa_snps_Cef.csv"
)
loss_evolution_cef <- loss_evolution_cef %>%
    filter(Epoch >= 2)
#start from the third epoch for plottability

loss_evolution_cip <- read_csv(
    "pipelines/nn_trained_models/intermediate_fusion_rf/loss_function/intermediate_fusion_rf_genexp_gpa_snps_Cip.csv"
)
loss_evolution_cip <- loss_evolution_cip %>%
    filter(Epoch >= 2)

loss_evolution_mer <- read_csv(
    "pipelines/nn_trained_models/intermediate_fusion_rf/loss_function/intermediate_fusion_rf_genexp_gpa_snps_Mer.csv"
)
loss_evolution_mer <- loss_evolution_mer %>%
    filter(Epoch >= 2)

loss_evolution_tob <- read_csv(
    "pipelines/nn_trained_models/intermediate_fusion_rf/loss_function/intermediate_fusion_rf_genexp_gpa_snps_Tob.csv"
)
loss_evolution_tob <- loss_evolution_tob %>%
    filter(Epoch >= 2)


plot <- ggplot() +
    geom_line(data = loss_evolution_cef, aes(x = Epoch, y = Loss, color = "Ceftazidim"), linewidth = 1) +
    geom_point(data = loss_evolution_cef, aes(x = Epoch, y = Loss, color = "Ceftazidim"), size = 2) +
    geom_line(data = loss_evolution_cip, aes(x = Epoch, y = Loss, color = "Ciprofloxacin"), linewidth = 1) +
    geom_point(data = loss_evolution_cip, aes(x = Epoch, y = Loss, color = "Ciprofloxacin"), size = 2) +
    geom_line(data = loss_evolution_mer, aes(x = Epoch, y = Loss, color = "Meropenem"), linewidth = 1) +
    geom_point(data = loss_evolution_mer, aes(x = Epoch, y = Loss, color = "Meropenem"), size = 2) +
    geom_line(data = loss_evolution_tob, aes(x = Epoch, y = Loss, color = "Tobramycin"), linewidth = 1) +
    geom_point(data = loss_evolution_tob, aes(x = Epoch, y = Loss, color = "Tobramycin"), size = 2) +
    labs(title = "Loss evolution over epochs intermediate fusion relevant features", x = "Epoch", y = "Loss", color = "Drug") +
    theme_bw()

ggsave("plots/figure_9/intermediate_fusion_rf.png", plot = plot, width = 6.67, height = 6.67)