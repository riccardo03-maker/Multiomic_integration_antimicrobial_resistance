library(tidyverse)
classes <- read_csv(
    "transformed_data/classes/classes.csv"
)

#count S vs R for each drug
Cef_S_vs_R <- classes %>%
    drop_na(Cef) %>%
    group_by(Cef) %>%
    summarize(Counts = n())

Cip_S_vs_R <- classes %>%
    drop_na(Cip) %>%
    group_by(Cip) %>%
    summarize(Counts = n())

Mer_S_vs_R <- classes %>%
    drop_na(Mer) %>%
    group_by(Mer) %>%
    summarize(Counts = n())

Tob_S_vs_R <- classes %>%
    drop_na(Tob) %>%
    group_by(Tob) %>%
    summarize(Counts = n())

colnames(Cef_S_vs_R) <- c("Class", "Counts")
colnames(Cip_S_vs_R) <- c("Class", "Counts")
colnames(Mer_S_vs_R) <- c("Class", "Counts")
colnames(Tob_S_vs_R) <- c("Class", "Counts")

#create a unique set
counts_data <- bind_rows(Cef_S_vs_R, Cip_S_vs_R, Mer_S_vs_R, Tob_S_vs_R)
counts_data <- counts_data %>%
    mutate(Drug = c("Cef", "Cef", "Cip", "Cip", "Mer", "Mer", "Tob", "Tob")) %>%
    mutate(Drug_class = c("Cef_Sus", "Cef_Res", "Cip_Sus", "Cip_Res", "Mer_Sus", "Mer_Res", "Tob_Sus", "Tob_Res"))

#plot
p <- ggplot(data = counts_data, aes(x = Drug_class, y = Counts, fill = Drug)) +
    geom_bar(stat = "identity") +
    labs(title = "Number of samples per class", x = "Drug and class", y = "Number of samples", fill = "Drug") +
    geom_text(aes(label = Counts), vjust = 2) +
    theme_gray()
ggsave("plots/figure_1.png", plot = p, width = 6.67, height = 6.67)