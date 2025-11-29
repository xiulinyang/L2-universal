library(tidyverse)
library(ggbreak)

files <- c(
  "/Users/xiulinyang/Desktop/TODO/L2-universal/results/results_B-GPT_el_en_sequential_all_checkpoints.csv",
  "/Users/xiulinyang/Desktop/TODO/L2-universal/results/results_B-GPT_es_en_sequential_all_checkpoints.csv",
  "/Users/xiulinyang/Desktop/TODO/L2-universal/results/results_B-GPT_nl_en_sequential_all_checkpoints.csv",
  "/Users/xiulinyang/Desktop/TODO/L2-universal/results/results_B-GPT_pl_en_sequential_all_checkpoints.csv"
)

get_lang <- function(path) {
  fn <- basename(path)
  str_extract(fn, "el_en|es_en|nl_en|pl_en")
}

df <- purrr::map_dfr(files, function(f) {
  readr::read_csv(f, show_col_types = FALSE) %>%
    dplyr::mutate(lang = get_lang(f))
})

df <- df %>%
  mutate(
    checkpoint = as.numeric(checkpoint),
    phenomenon = factor(
      phenomenon,
      levels = c("yn_questions", "wh_questions"),
      labels = c("YN questions", "WH questions")
    ),
    lang = factor(
      lang,
      levels = c("el_en", "es_en", "nl_en", "pl_en"),
      labels = c("EL–EN", "ES–EN", "NL–EN", "PL–EN")
    )
  ) %>%
  pivot_longer(
    c(s1_pref, s2_pref, s3_pref),
    names_to = "preference",
    values_to = "value"
  ) %>%
  mutate(
    preference = factor(
      preference,
      levels = c("s1_pref", "s2_pref", "s3_pref"),
      labels = c("S1", "S2", "S3")
    )
  )


df <- df %>%
  filter(!(phenomenon == "YN questions" & preference == "S3"))

df <- df %>%
  mutate(
    series = case_when(
      # YN
      phenomenon == "YN questions" & preference == "S1" ~ "YN S1: Victoria does stand?",
      phenomenon == "YN questions" & preference == "S2" ~ "YN S2: Does Victoria stand?",
      # WH
      phenomenon == "WH questions" & preference == "S1" ~ "WH S1: the french actor does marry who?",
      phenomenon == "WH questions" & preference == "S2" ~ "WH S2: who the french actor does marry?",
      phenomenon == "WH questions" & preference == "S3" ~ "WH S3: who does the french actor marry?",
      TRUE ~ NA_character_
    ),
    series = factor(
      series,
      levels = c(
        "YN S1: Victoria does stand?",
        "YN S2: Does Victoria stand?",
        "WH S1: the french actor does marry who?",
        "WH S2: who the french actor does marry?",
        "WH S3: who does the french actor marry?"
      )
    )
  )

library(RColorBrewer)
cols <- brewer.pal(5, "Set1")

series_colors <- c(
  "YN S1: Victoria does stand?"              = "#22763F",
  "YN S2: Does Victoria stand?"              = "#D44D44",
  "WH S1: the french actor does marry who?"  = "#699CC5",
  "WH S2: who the french actor does marry?"  = "#F6BD5E",
  "WH S3: who does the french actor marry?"  = "#D44D44"
)


p_full <- ggplot(
  df,
  aes(
    x = checkpoint,
    y = value,
    color = series,                                  
    group = interaction(lang, phenomenon, preference) 
  )
) +
  geom_line(linewidth = 1.0) +
  geom_vline(xintercept = 64000, linetype = "dashed") +
  facet_grid(
    rows = vars(lang),
    cols = vars(phenomenon)
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(
    values = series_colors,
    name   = "Sentence"
  ) +
  labs(
    x = "Checkpoint",
    y = "Preference score",
    title = ""
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "grey90"),
    panel.grid.minor = element_blank()
  )

p_full


df_zoom <- df %>%
  filter(checkpoint >= 63000, checkpoint <= 65000)

p_zoom <- ggplot(
  df_zoom,
  aes(
    x = checkpoint,
    y = value,
    color = series,
    group = interaction(lang, phenomenon, preference)
  )
) +
  geom_line(linewidth = 1.0) +
  geom_vline(xintercept = 64000, linetype = "dashed") +
  facet_grid(
    rows = vars(lang),
    cols = vars(phenomenon)
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(
    values = series_colors,
    name   = "Sentence"
  ) +
  labs(
    x = "Checkpoint (63000–65000)",
    y = "Preference score",
    title = ""
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "grey90"),
    panel.grid.minor = element_blank()
  )

p_zoom