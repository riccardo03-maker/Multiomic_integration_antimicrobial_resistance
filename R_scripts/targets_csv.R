library(tidyverse)

targets <- read_csv(
    "./transformed_data/targets/phenotypes.txt"
)

#remove colistin column, not used in analysis
targets <- targets %>%
    select(!starts_with("Colistin"))

#change column names: Strain, Tob, Cef, Cip, Mer
colnames(targets) <- c("Strain", "Tob", "Cef", "Cip", "Mer")

write_csv(targets, "./transformed_data/targets/targets.csv")