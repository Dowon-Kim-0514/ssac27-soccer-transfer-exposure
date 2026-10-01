root <- normalizePath(commandArgs(trailingOnly=TRUE)[1])
library(worldfootballR)
library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(ggrepel)
library(patchwork)
library(stringdist)

options(timeout=45)
seasons <- c(2018, 2019, 2020, 2021, 2022, 2023, 2024)

all_data <- list()
for (stat in c("shooting", "passing", "possession", "misc")) {
  cat("수집 중:", stat, "\n")
  temp <- load_fb_big5_advanced_season_stats(
    season_end_year = seasons,
    stat_type = stat,
    team_or_player = "player"
  )
  all_data[[stat]] <- temp
  Sys.sleep(2)
}

standard_data <- load_fb_big5_advanced_season_stats(
  season_end_year = seasons,
  stat_type = "standard",
  team_or_player = "player"
)

defense_data <- load_fb_big5_advanced_season_stats(
  season_end_year = seasons,
  stat_type = "defense",
  team_or_player = "player"
)

keepers_data <- load_fb_big5_advanced_season_stats(
  season_end_year = seasons,
  stat_type = "keepers",
  team_or_player = "player"
)

keepers_adv_data <- load_fb_big5_advanced_season_stats(
  season_end_year = seasons,
  stat_type = "keepers_adv",
  team_or_player = "player"
)

shooting   <- all_data[["shooting"]]
passing    <- all_data[["passing"]]
possession <- all_data[["possession"]]
misc       <- all_data[["misc"]]

standard_clean <- standard_data %>%
  select(-xG_Expected, -npxG_Expected)

shooting_clean <- shooting %>%
  select(Player, Squad, Season_End_Year, xG_Expected, npxG_Expected)

passing_clean <- passing %>%
  select(Player, Squad, Season_End_Year, Cmp_percent_Total, KP, PrgP)

possession_clean <- possession %>%
  select(Player, Squad, Season_End_Year,
         PrgC_Carries, PrgDist_Carries, Final_Third_Carries, Succ_Take)

misc_clean <- misc %>%
  select(Player, Squad, Season_End_Year,
         Int, TklW, Won_percent_Aerial, Recov)

defense_clean <- defense_data %>%
  select(Player, Squad, Season_End_Year,
         Tkl_Tackles, "Tkl+Int", Clr, Blocks_Blocks)

keepers_clean <- keepers_data %>%
  select(Player, Squad, Season_End_Year,
         Save_percent, GA90, CS_percent)

keepers_adv_clean <- keepers_adv_data %>%
  select(Player, Squad, Season_End_Year,
         PSxG = PSxG_Expected)

standard_npgoals <- standard_data %>%
  select(Player, Squad, Season_End_Year, G_minus_PK)

combined_data <- standard_clean %>%
  left_join(shooting_clean,
            by = c("Player", "Squad", "Season_End_Year")) %>%
  left_join(passing_clean,
            by = c("Player", "Squad", "Season_End_Year")) %>%
  left_join(possession_clean,
            by = c("Player", "Squad", "Season_End_Year")) %>%
  left_join(misc_clean,
            by = c("Player", "Squad", "Season_End_Year")) %>%
  left_join(defense_clean,
            by = c("Player", "Squad", "Season_End_Year")) %>%
  left_join(keepers_clean,
            by = c("Player", "Squad", "Season_End_Year")) %>%
  left_join(keepers_adv_clean,
            by = c("Player", "Squad", "Season_End_Year")) %>%
  left_join(standard_npgoals,
            by = c("Player", "Squad", "Season_End_Year")) %>%
  mutate(transfer_season = case_when(
    Season_End_Year == 2018 ~ "17/18",
    Season_End_Year == 2019 ~ "18/19",
    Season_End_Year == 2020 ~ "19/20",
    Season_End_Year == 2021 ~ "20/21",
    Season_End_Year == 2022 ~ "21/22",
    Season_End_Year == 2023 ~ "22/23",
    Season_End_Year == 2024 ~ "23/24"
  ))

# 중복 제거 (시즌 중 이적 선수 — 출전시간 최대값 선택)
fbref_dedup <- combined_data %>%
  group_by(Player, transfer_season) %>%
  slice_max(Mins_Per_90_Playing, n = 1, with_ties = FALSE) %>%
  ungroup()

write_csv(combined_data, file.path(root, "data", "work", "acquisition", "fbref_combined.csv"))
cat("FBref 저장 완료:", nrow(combined_data), "행 /",
    ncol(combined_data), "컬럼\n")
cat("FBref 중복 제거 후:", nrow(fbref_dedup), "행\n")

dir.create(file.path(root, "data", "work", "acquisition", "downloaded_tables"), showWarnings=FALSE)
for (name in c("standard_data", "shooting", "passing", "possession", "misc", "defense_data", "keepers_data", "keepers_adv_data")) {
  write_csv(get(name), file.path(file.path(root, "data", "work", "acquisition", "downloaded_tables"), paste0(name, ".csv")))
}
capture.output(sessionInfo(), file=file.path(root, "data", "work", "acquisition", "R_sessionInfo.txt"))
