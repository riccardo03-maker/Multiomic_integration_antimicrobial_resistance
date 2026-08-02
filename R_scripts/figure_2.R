#load pca results
library(tidyverse)
pca_cef <- read_csv(
    "pipelines/results/pca/pca_Cef.csv"
)
pca_cip <- read_csv(
    "pipelines/results/pca/pca_Cip.csv"
)
pca_mer <- read_csv(
    "pipelines/results/pca/pca_Mer.csv"
)
pca_tob <- read_csv(
    "pipelines/results/pca/pca_Tob.csv"
)

# create a plot for each feature type and each drug
pca_cef_genexp <- ggplot(data = pca_cef, aes(x = genexp_1, y = genexp_2)) +
    geom_point(aes(color = as.character(Cef))) +
    labs(title = "PCA Ceftazidim genexp", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_cef_genexp.png", plot = pca_cef_genexp, width = 6.67, height = 6.67)

pca_cip_genexp <- ggplot(data = pca_cip, aes(x = genexp_1, y = genexp_2)) +
    geom_point(aes(color = as.character(Cip))) +
    labs(title = "PCA Ciprofloxacin genexp", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_cip_genexp.png", plot = pca_cip_genexp, width = 6.67, height = 6.67)

pca_mer_genexp <- ggplot(data = pca_mer, aes(x = genexp_1, y = genexp_2)) +
    geom_point(aes(color = as.character(Mer))) +
    labs(title = "PCA Meropenem genexp", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_mer_genexp.png", plot = pca_mer_genexp, width = 6.67, height = 6.67)

pca_tob_genexp <- ggplot(data = pca_tob, aes(x = genexp_1, y = genexp_2)) +
    geom_point(aes(color = as.character(Tob))) +
    labs(title = "PCA Tobramycin genexp", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_tob_genexp.png", plot = pca_tob_genexp, width = 6.67, height = 6.67)

pca_cef_gpa <- ggplot(data = pca_cef, aes(x = gpa_1, y = gpa_2)) +
    geom_point(aes(color = as.character(Cef))) +
    labs(title = "PCA Ceftazidim gpa", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_cef_gpa.png", plot = pca_cef_gpa, width = 6.67, height = 6.67)

pca_cip_gpa <- ggplot(data = pca_cip, aes(x = gpa_1, y = gpa_2)) +
    geom_point(aes(color = as.character(Cip))) +
    labs(title = "PCA Ciprofloxacin gpa", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_cip_gpa.png", plot = pca_cip_gpa, width = 6.67, height = 6.67)

pca_mer_gpa <- ggplot(data = pca_mer, aes(x = gpa_1, y = gpa_2)) +
    geom_point(aes(color = as.character(Mer))) +
    labs(title = "PCA Meropenem gpa", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_mer_gpa.png", plot = pca_mer_gpa, width = 6.67, height = 6.67)

pca_tob_gpa <- ggplot(data = pca_tob, aes(x = gpa_1, y = gpa_2)) +
    geom_point(aes(color = as.character(Tob))) +
    labs(title = "PCA Tobramycin gpa", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_tob_gpa.png", plot = pca_tob_gpa, width = 6.67, height = 6.67)

pca_cef_snps <- ggplot(data = pca_cef, aes(x = snps_1, y = snps_2)) +
    geom_point(aes(color = as.character(Cef))) +
    labs(title = "PCA Ceftazidim snps", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_cef_snps.png", plot = pca_cef_snps, width = 6.67, height = 6.67)

pca_cip_snps <- ggplot(data = pca_cip, aes(x = snps_1, y = snps_2)) +
    geom_point(aes(color = as.character(Cip))) +
    labs(title = "PCA Ciprofloxacin snps", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_cip_snps.png", plot = pca_cip_snps, width = 6.67, height = 6.67)

pca_mer_snps <- ggplot(data = pca_mer, aes(x = snps_1, y = snps_2)) +
    geom_point(aes(color = as.character(Mer))) +
    labs(title = "PCA Meropenem snps", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_mer_snps.png", plot = pca_mer_snps, width = 6.67, height = 6.67)

pca_tob_snps <- ggplot(data = pca_tob, aes(x = snps_1, y = snps_2)) +
    geom_point(aes(color = as.character(Tob))) +
    labs(title = "PCA Tobramycin snps", x = "Component 1", y = "Component 2", color = "S-vs-R")
ggsave("plots/figure_2/pca_tob_snps.png", plot = pca_tob_snps, width = 6.67, height = 6.67)