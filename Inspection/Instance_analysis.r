require(tidyverse)
require(rstudioapi)
require(ggpubr)
require(reshape2)
require(viridis)
require(jsonlite)

## Settings
# for linux full path might be necessary due to image export from EBImage that can't deal with relative path
# RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/BriFi', '~/Documents/BriFi')
# RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/GitHub/BriFi', '~/Documents/GitHub/BriFi')
# ## set direction to find predictions and to export instances
# dataset = "Task001_A549"
# run = 2
# fold = 5
if (is.null(fold)) {
    Save_dir = paste0(ifelse(grepl("Windows", sessionInfo()$running), '~/BF_Data/', '~/Documents/BF_Data/'), dataset, "/Prediction/Run--", run)
    source(paste0(Save_dir, "/Params.r"))
    Save_plot_instance = paste0(Save_dir, "/Ensemble_Plot_instance")
    dir.create(Save_plot_instance, showWarnings = F)
} else {
    Save_dir = paste0(ifelse(grepl("Windows", sessionInfo()$running), '~/BF_Data/', '~/Documents/BF_Data/'), dataset, "/Prediction/Run--", run)
    source(paste0(Save_dir, "/Params.r"))
    Save_dir = paste0(ifelse(grepl("Windows", sessionInfo()$running), '~/BF_Data/', '~/Documents/BF_Data/'), dataset, "/Prediction/Run--", run, "/FOLD_", fold)
    Save_plot_instance = paste0(Save_dir, "/Plot_instance")
    dir.create(Save_plot_instance, showWarnings = F)
}

# bind all data together 
data = tibble()

list_data = list.files(path = paste0(Save_dir, "/data"), pattern = ".RData", full.names = TRUE)
message(paste0("Load data from predicted instances in dataset ", dataset))

for(i in list_data){
    print(i)
    load(i)
    temp = get("single_dice_val")  %>%
        mutate(file_name = i,
                file_name = sub(".*/data/", "", file_name),
                # TASK = as.numeric(TASK),
                DICE_ID = sub(".*single_dice_", "", file_name),
                DICE_ID = as.numeric(sub("_ID.*", "", DICE_ID)),
                ID = sub(".*_ID--", "", file_name),
                ID = as.numeric(sub("_WELL.*", "", ID)),
                # ID = map2(ID, TASK, ~ ifelse(.y == "006", .x = .x - 224, .x)))#, # ID for Task006 have an offset despite same source images
                # ID = map_if(ID, TASK == 6, ~ .x - 224),
                ID = unlist(ID),
                # ID = map_dbl(ID),
                # ID = unnest(ID),
                WELL = sub(".*_WELL--", "", file_name),
                WELL = sub("\\.RData.*", "", WELL)
                )
    
    temp = temp %>% 
        select(-c(file_name)) %>% 
        select(DICE_ID, ID, WELL, everything())
    data = rbind(data, temp)
}

data_qc = data %>%
     mutate(extra = case_when(is.na(inst_true) & extra == 0 ~ 1, TRUE ~ extra),
            undersplit = abs(undersplit))  # to rectify for extra nuclei in pred absent in gt

data_qc_sum = data_qc %>%
    group_by(DICE_ID) %>%
    summarise(dice = mean(dice_score, na.rm = TRUE),
            #   count_gt = sum(!is.na(inst_true)),
            #   count_pred = sum(!is.na(inst_pred)),
              count_gt = max(inst_true),
              count_pred = max(inst_pred),
              undersplit = sum(undersplit, na.rm = TRUE)/n(),
              oversplit = sum(oversplit, na.rm = TRUE)/n(),
              extra = sum(extra, na.rm = TRUE)/n(),
              missed = sum(missed, na.rm = TRUE)/n()
              ) %>% 
    mutate(count = count_pred / count_gt,
           class = CLASS,
           strategy = NA,
           strategy = map2(strategy, class, ~ ifelse(.y == 1, paste(ARCHITECTURE, "\n\n", ENCODER, "\n\n", "WS", sep=""), .x)),
           strategy = map2(strategy, class, ~ ifelse(.y == 2, paste(ARCHITECTURE, "\n\n", ENCODER, "\n\n", "CCA", sep=""), .x)),
           strategy = unlist(strategy)
           ) %>%
    select(class, strategy, dice, count, everything())

######################################################################################################################
####################################     Summary per metric to JSON     ##############################################
######################################################################################################################

json_metric = list()

json_metric$Test = I(list(
    `dice` = data_qc_sum %>% select(dice) %>% pull %>% mean,
    `relative_count` = data_qc_sum %>% select(count) %>% pull %>% mean,
    `count_gt` = data_qc_sum %>% select(count_gt) %>% pull %>% mean,
    `count_pred` = data_qc_sum %>% select(count_pred) %>% pull %>% mean,
    `missed` = data_qc_sum %>% select(missed) %>% pull %>% mean,
    `extra` = data_qc_sum %>% select(extra) %>% pull %>% mean,
    `oversplit` = data_qc_sum %>% select(oversplit) %>% pull %>% mean,
    `undersplit` = data_qc_sum %>% select(undersplit) %>% pull %>% mean
    ))

json_metric = toJSON(json_metric, auto_unbox = TRUE, pretty = TRUE)
write_json_data_index(json_metric, paste0(Save_dir, "/detail_metric_instance_",
                                          ifelse(is.null(fold), "ensemble", paste0("fold_", fold)), ".json"))

######################################################################################################################
####################################     One boxplot per metric     ##################################################
######################################################################################################################

for(plt in c("dice", "count", "count_gt", "count_pred", # "count_gt_rel", "count_pred_rel",
             "missed", "extra", "oversplit", "undersplit")){

    plot = data_qc_sum %>% 
        ungroup() %>%
        mutate(Strategy = strategy) %>%
        ggboxplot(x = "Strategy",
                  y = plt,
                  desc_stat = "mean_sd",
                  xlab = "") + #,
                  #facet.by = "Drug") +
        theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5))

    ggsave(paste0(Save_plot_instance, "/", plt, ".png"), plot, width = 4, height = 4)

}

######################################################################################################################
####################################     Panel of metric together   ##################################################
######################################################################################################################

data_qc_sum_resh = as_tibble(melt(data_qc_sum %>% ungroup %>% select(-c(class, DICE_ID, count_gt, count_pred)), id.vars = c("strategy")))
data_qc_sum_resh %>% select(variable) %>% pull %>% unique

plot_gr1 = data_qc_sum_resh %>% 
        filter(variable %in% c("dice",
                               "count"
                            #    "count"
                               )) %>%
        mutate(variable = case_when(variable == "dice" ~ "Dice",
                                    variable == "count" ~ "Count")#,
                        #  variable == "count_gt_rel" ~ "Rel. true cell count",
                        #  variable == "count_pred_rel" ~ "Rel. pred. cell count")
                         ) %>%
        # mutate(Concentration = round(Concentration, 2)) %>%
        mutate(variable = factor(variable, levels = c("Dice", "Count"))) %>% 
        ggerrorplot(x = "strategy",
                    y = "value",
                    color = "variable",
                    desc_stat = "mean_sd",
                    # facet.by = "Drug",
                    legend.title = "Measure",
                    xlab = "",
                    ylab = "") + #,
                    # palette = c("#00AFBB", "#E7B800", "#FC4E07")) +
        theme_classic2() +
        theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5)) +
        scale_color_viridis(discrete = TRUE, end = 0.5) +
        scale_y_continuous(breaks = get_breaks(n = 11)) +
        theme(
        panel.background = element_rect(fill = NA),
        panel.grid.major = element_line(colour = "grey90"),
        panel.ontop = FALSE
        ) + 
        theme(legend.position="top")
        # scale_color_grey(start = 0.7, end = 0.3) 
ggsave(paste0(Save_plot_instance, "/", "pltgrp1_dice_rel_count", ".png"), plot_gr1, width = 5, height = 4)

# library(ggbreak)
plot_gr2 = data_qc_sum_resh %>% 
        filter(variable %in% c("extra",
                               "missed",
                               "undersplit",
                               "oversplit"
                               )) %>%
        mutate(variable = case_when(variable == "extra" ~ "Extra",
                         variable == "missed" ~ "Missed",
                         variable == "undersplit" ~ "Undersplit",
                         variable == "oversplit" ~ "Oversplit")) %>%
        # mutate(Concentration = round(Concentration, 2)) %>%
        # mutate(Drug = factor(Drug, levels = c("Vehicle", "Staurosporin", "Doxorubicin", "Nocodazole", "Vorinostat", "Nigericin"))) %>% 
        ggerrorplot(x = "strategy",
                    y = "value",
                    color = "variable",
                    desc_stat = "mean_sd",
                    # facet.by = "Drug",
                    legend.title = "Measure",
                    xlab = "",
                    ylab = ""
                    ) +#,
        # scale_y_break(c(0.2, 0.5)) +
                    # palette = c("#00AFBB", "#E7B800", "#FC4E07")) +
        theme_classic2() +
        theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust = 0.5)) +
        scale_color_viridis(discrete = TRUE) +         
        scale_y_continuous(breaks = get_breaks(n = 10)) +
        theme(
        panel.background = element_rect(fill = NA),
        panel.grid.major = element_line(colour = "grey90"),
        panel.ontop = FALSE
        ) + 
        theme(legend.position="top")
        # scale_color_grey(start = 0.7, end = 0.3) 
ggsave(paste0(Save_plot_instance, "/", "pltgrp2_extra_missed_over_under", ".png"), plot_gr2, width = 5, height = 4)