library(worldfootballR)
library(dplyr)
library(tidyr)
library(ggplot2)
library(readr)
library(ggrepel)
library(patchwork)
library(stringdist)

combined_data <- readr::read_csv(file.path(root, "data", "inputs", "reference", "fbref_combined.csv"))
fbref_dedup <- combined_data %>%
  group_by(Player, transfer_season) %>%
  slice_max(Mins_Per_90_Playing, n = 1, with_ties = FALSE) %>%
  ungroup()

write_csv(combined_data, file.path(root, "data", "work", "acquisition", "fbref_combined.csv"))
cat("FBref 저장 완료:", nrow(combined_data), "행 /",
    ncol(combined_data), "컬럼\n")
cat("FBref 중복 제거 후:", nrow(fbref_dedup), "행\n")

transfers <- read_csv(file.path(root, "data", "inputs", "kaggle", "transfers.csv"))
players   <- read_csv(file.path(root, "data", "inputs", "kaggle", "players.csv"))
clubs     <- read_csv(file.path(root, "data", "inputs", "kaggle", "clubs.csv"))

big5_ids <- c("GB1", "ES1", "L1", "IT1", "FR1")

big5_clubs <- clubs %>%
  filter(domestic_competition_id %in% big5_ids) %>%
  select(club_id, name, domestic_competition_id)

transfer_clean <- transfers %>%
  left_join(players %>% select(player_id, name, position, sub_position,
                                date_of_birth, contract_expiration_date,
                                country_of_citizenship, height_in_cm),
            by = "player_id") %>%
  filter(transfer_season %in% c("17/18","18/19","19/20","20/21",
                                  "21/22","22/23","23/24")) %>%
  filter(transfer_fee > 0) %>%
  mutate(
    age_at_transfer = as.numeric(difftime(transfer_date,
                                           date_of_birth,
                                           units = "days")) / 365.25
  )

transfer_big5 <- transfer_clean %>%
  filter(to_club_id %in% big5_clubs$club_id) %>%
  left_join(big5_clubs %>% rename(to_league = domestic_competition_id,
                                   to_club_name_full = name),
            by = c("to_club_id" = "club_id")) %>%
  left_join(big5_clubs %>% rename(from_league = domestic_competition_id,
                                   from_club_name_full = name),
            by = c("from_club_id" = "club_id"))

# 같은 시즌 두 번 이적한 경우 이적료 최대값 선택
transfer_big5_dedup <- transfer_big5 %>%
  group_by(player_name, transfer_season) %>%
  slice_max(transfer_fee, n = 1, with_ties = FALSE) %>%
  ungroup()

write_csv(transfer_big5, file.path(root, "data", "work", "acquisition", "transfer_big5.csv"))
cat("Transfermarkt 저장 완료:", nrow(transfer_big5_dedup), "행\n")

unmatched_all <- transfer_big5_dedup %>%
  left_join(fbref_dedup,
            by = c("player_name" = "Player",
                   "transfer_season" = "transfer_season")) %>%
  filter(is.na(xG_Expected)) %>%
  select(player_name, transfer_season) %>%
  distinct()

fbref_players <- fbref_dedup %>%
  select(Player, transfer_season) %>%
  distinct()

fuzzy_all <- unmatched_all %>%
  rowwise() %>%
  mutate(
    best_match = fbref_players$Player[
      which.min(stringdist(player_name, fbref_players$Player, method = "jw"))
    ],
    match_score = min(stringdist(player_name, fbref_players$Player, method = "jw"))
  ) %>%
  ungroup() %>%
  filter(match_score < 0.1) %>%
  filter(player_name != best_match) %>%
  filter(!(player_name == "Dani Silva" & best_match == "David Silva")) %>%
  filter(!(player_name == "Gabriel Moscardo" & best_match == "Gabriel Mercado"))

cat("이름 교체 대상:", nrow(fuzzy_all), "건\n")

transfer_big5_fixed <- transfer_big5_dedup %>%
  left_join(fuzzy_all %>% select(player_name, transfer_season, best_match),
            by = c("player_name", "transfer_season")) %>%
  mutate(player_name = ifelse(!is.na(best_match), best_match, player_name)) %>%
  select(-best_match) %>%
  mutate(prev_season = case_when(
    transfer_season == "17/18" ~ "17/18",
    transfer_season == "18/19" ~ "17/18",
    transfer_season == "19/20" ~ "18/19",
    transfer_season == "20/21" ~ "19/20",
    transfer_season == "21/22" ~ "20/21",
    transfer_season == "22/23" ~ "21/22",
    transfer_season == "23/24" ~ "22/23"
  ))

final_prev <- transfer_big5_fixed %>%
  left_join(fbref_dedup,
            by = c("player_name" = "Player",
                   "prev_season" = "transfer_season"))

cat("전체 이적 건수:", nrow(transfer_big5_fixed), "\n")
cat("FBref 매칭 성공:", sum(!is.na(final_prev$xG_Expected)), "\n")
cat("FBref 매칭 실패:", sum(is.na(final_prev$xG_Expected)), "\n")

uefa_coeff <- c(
  "GB1" = 90.250,
  "ES1" = 82.550,
  "L1"  = 72.850,
  "IT1" = 71.050,
  "FR1" = 59.650
)

league_level <- c(
  "GB1" = 5,
  "ES1" = 4,
  "L1"  = 3,
  "IT1" = 2,
  "FR1" = 1
)

final_clean <- final_prev %>%
  filter(!is.na(xG_Expected)) %>%
  rename(npGoals = G_minus_PK.x) %>%
  mutate(
    age_at_transfer   = floor(age_at_transfer),
    transfer_fee_raw  = transfer_fee,
    transfer_fee      = paste0("$", formatC(transfer_fee, format = "f",
                                             digits = 0, big.mark = ",")),
    transfer_season   = case_when(
      transfer_season == "17/18" ~ "2017-18",
      transfer_season == "18/19" ~ "2018-19",
      transfer_season == "19/20" ~ "2019-20",
      transfer_season == "20/21" ~ "2020-21",
      transfer_season == "21/22" ~ "2021-22",
      transfer_season == "22/23" ~ "2022-23",
      transfer_season == "23/24" ~ "2023-24"
    ),
    covid_dummy              = ifelse(transfer_season %in%
                                        c("2019-20", "2020-21"), 1, 0),
    contract_years_remaining = as.numeric(
      difftime(as.Date(contract_expiration_date),
               as.Date(paste0(Season_End_Year, "-07-01")),
               units = "days")) / 365.25,
    log_transfer_fee  = log(transfer_fee_raw),

    # ── [v5.0 수정] U21_dummy: 만 21세 이하 = 1 (기존 < 21 에서 <= 21 로 수정) ──
    U21_dummy         = ifelse(age_at_transfer <= 21, 1, 0),

    # ── [v5.0 신규] O30_dummy: 만 30세 이상 = 1 ──────────────────────────────
    O30_dummy         = ifelse(age_at_transfer >= 30, 1, 0),

    age_squared       = age_at_transfer^2,
    season_proxy_flag = ifelse(transfer_season == "2017-18", 1, 0),
    UEFA_coeff_from   = uefa_coeff[from_league],
    league_level_diff = case_when(
      is.na(from_league) ~ 0,
      league_level[to_league] > league_level[from_league]  ~  1,
      league_level[to_league] == league_level[from_league] ~  0,
      league_level[to_league] < league_level[from_league]  ~ -1
    ),
    # per 90 변환
    Gls_per90         = Gls / Mins_Per_90_Playing,
    Ast_per90         = Ast / Mins_Per_90_Playing,
    GA_per90          = `G+A` / Mins_Per_90_Playing,
    npGoals_per90     = npGoals / Mins_Per_90_Playing,
    PrgC_per90        = PrgC_Carries / Mins_Per_90_Playing,
    PrgDist_per90     = PrgDist_Carries / Mins_Per_90_Playing,
    Final_Third_per90 = Final_Third_Carries / Mins_Per_90_Playing,
    PrgP_per90        = PrgP / Mins_Per_90_Playing,
    Succ_Take_per90   = Succ_Take / Mins_Per_90_Playing,
    KP_per90          = KP / Mins_Per_90_Playing,
    Tkl_per90         = Tkl_Tackles / Mins_Per_90_Playing,
    TklInt_per90      = `Tkl+Int` / Mins_Per_90_Playing,
    Int_per90         = Int / Mins_Per_90_Playing,
    TklW_per90        = TklW / Mins_Per_90_Playing,
    Clr_per90         = Clr / Mins_Per_90_Playing,
    Blocks_per90      = Blocks_Blocks / Mins_Per_90_Playing,
    Recov_per90       = Recov / Mins_Per_90_Playing
  ) %>%
  select(
    # 선수 기본 정보
    player_name, transfer_season, to_league, from_league,
    position, sub_position, age_at_transfer, age_squared, height_in_cm,
    # 이적 정보
    from_club_name, to_club_name_full,
    transfer_fee, transfer_fee_raw, log_transfer_fee,
    market_value_in_eur, contract_expiration_date, contract_years_remaining,
    # 구조적 통제 변수 — [v5.0] O30_dummy 추가
    covid_dummy, U21_dummy, O30_dummy, season_proxy_flag,
    UEFA_coeff_from, league_level_diff,
    # 출전 시간
    Mins_Per_90_Playing,
    # 전통 지표 (절대값 + per90)
    Gls, Ast, `G+A`, npGoals,
    Gls_per90, Ast_per90, GA_per90, npGoals_per90,
    # xG 기반 지표 (이미 per90)
    xG_Expected, npxG_Expected, xAG_Expected, `npxG+xAG_Expected`,
    xG_Per, xAG_Per, `npxG+xAG_Per`,
    # 볼 진전 (절대값 + per90)
    PrgC_Carries, PrgDist_Carries, Final_Third_Carries, PrgP,
    PrgC_per90, PrgDist_per90, Final_Third_per90, PrgP_per90,
    # 개인 돌파 + 패스 (절대값 + per90)
    Succ_Take, KP, Cmp_percent_Total,
    Succ_Take_per90, KP_per90,
    # 수비 (절대값 + per90)
    Tkl_Tackles, `Tkl+Int`, Int, TklW, Clr, Blocks_Blocks,
    Won_percent_Aerial, Recov,
    Tkl_per90, TklInt_per90, Int_per90, TklW_per90,
    Clr_per90, Blocks_per90, Recov_per90,
    # GK 전용
    Save_percent, GA90, CS_percent, PSxG,
    Season_End_Year
  )

# 최종 확인
cat("최종 데이터:", nrow(final_clean), "건\n")
cat("컬럼 수:", ncol(final_clean), "\n")
cat("\nU21_dummy 분포 (만 21세 이하):\n"); print(table(final_clean$U21_dummy))
cat("\nO30_dummy 분포 (만 30세 이상):\n"); print(table(final_clean$O30_dummy))

write_csv(final_clean, file.path(root, "data", "work", "acquisition", "clean_data.csv"))
cat("\nclean_data.csv 저장 완료\n")

write_csv(fuzzy_all, file.path(root, "data", "work", "acquisition", "reproduced_name_matches.csv"))

stages <- c("transfers", "players", "clubs", "combined_data", "fbref_dedup", "big5_clubs", "transfer_clean", "transfer_big5", "transfer_big5_dedup", "unmatched_all", "fuzzy_all", "final_prev", "final_clean")
write_csv(tibble(stage=stages, rows=sapply(stages, function(x) nrow(get(x))), columns=sapply(stages, function(x) ncol(get(x)))), file.path(root, "data", "work", "acquisition", "stage_counts.csv"))
write_csv(final_prev %>% select(player_id, player_name, transfer_date, transfer_season, prev_season, Season_End_Year, contract_expiration_date, xG_Expected), file.path(root, "data", "work", "acquisition", "matching_provenance.csv"))
capture.output(sessionInfo(), file=file.path(root, "data", "work", "acquisition", "R_sessionInfo.txt"))
