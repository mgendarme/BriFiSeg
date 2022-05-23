library(tidyverse)
library(rstudioapi)
library(ggpubr)
library(reshape2)
library(viridis)

## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
# TASK = 17
nnunetpath = str_replace(RelPath, "UNet", "nnUNet")
Save_dir = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Instance_comparison")

# set saving directories for plots 
# Save_data = paste0(Save_dir, "/data")
Save_plot_instance = paste0(Save_dir, "/Plot_instance")
dir.create(Save_plot_instance, showWarnings = F)
Tasks = c("001",    # A549  1 class unet plain          ws
          "002",    # A549  2 class unet plain          cca
          "006",    # A549  1 class unet seresnext101   ws
          "019")    # A549  2 class unet seresnext101   cca
# bind all data together 
data = tibble()

for(TASK in Tasks){
    list_data = list.files(path = paste0(Save_dir, "/Task", TASK, "/data"), pattern = ".RData", full.names = TRUE)
    message(paste0("Load data Task", TASK))
    for(i in list_data){
        print(i)
        load(i)
        temp = get("single_dice_val")  %>%
            mutate(file_name = i,
                # file_name = str_replace(file_name, "_ctr", ""),
                # file_name = str_replace(file_name, "_ts", ""),
                file_name = sub(".*/data/", "", file_name),
                TASK = as.numeric(TASK),
                DICE_ID = sub(".*single_dice_", "", file_name),
                DICE_ID = as.numeric(sub("_ID.*", "", DICE_ID)),
                ID = sub(".*_ID--", "", file_name),
                ID = as.numeric(sub("_WELL.*", "", ID)),
                # ID = map2(ID, TASK, ~ ifelse(.y == "006", .x = .x - 224, .x)))#, # ID for Task006 have an offset despite same source images
                ID = map_if(ID, TASK == 6, ~ .x - 224),
                ID = unlist(ID),
                # ID = map_dbl(ID),
                # ID = unnest(ID),
                WELL = sub(".*_WELL--", "", file_name),
                WELL = sub("\\.RData.*", "", WELL)
                )
        
        temp = temp %>% 
            select(-c(file_name)) %>% 
            select(TASK, DICE_ID, ID, WELL, everything())
        data = rbind(data, temp)
    }
}

data %>% select(ID) %>% pull %>% unique
data %>% select(DICE_ID) %>% pull %>% unique
data_bu = data # helps for testing to get back original data

data_qc = data %>%
     mutate(extra = case_when(is.na(inst_true) & extra == 0 ~ 1, TRUE ~ extra),
            undersplit = abs(undersplit))  # to rectify for extra nuclei in pred absent in gt
data_qc %>% 
     group_by(TASK) %>%
     summarise(count_gt = sum(!is.na(inst_true))) %>%
     select(count_gt) %>% pull

data_qc_sum = data_qc %>%
    group_by(TASK, DICE_ID) %>%
    summarise(dice = mean(dice_score, na.rm = TRUE),
              count_gt = sum(!is.na(inst_true)),
              count_pred = sum(!is.na(inst_pred)),
              undersplit = sum(undersplit, na.rm = TRUE)/n(),
              oversplit = sum(oversplit, na.rm = TRUE)/n(),
              extra = sum(extra, na.rm = TRUE)/n(),
              missed = sum(missed, na.rm = TRUE)/n()
              ) %>% 
    mutate(count = count_pred / count_gt,
           task = TASK,
           strategy = NA,
           strategy = map2(strategy, task, ~ ifelse(.y == 1, paste("U-Net", "\n\n", "/", "\n\n", "WS", sep=""), .x)),
           strategy = map2(strategy, task, ~ ifelse(.y == 2, paste("U-Net", "\n\n", "/", "\n\n", "CCA", sep=""), .x)),
           strategy = map2(strategy, task, ~ ifelse(.y == 6, paste("U-Net", "\n\n", "SE-101", "\n\n", "WS", sep=""), .x)),
           strategy = map2(strategy, task, ~ ifelse(.y == 19, paste("U-Net", "\n\n", "SE-101", "\n\n", "CCA", sep=""), .x)),
           strategy = unlist(strategy)
           ) %>%
    select(task, strategy, dice, count, everything(), -TASK)
    # select(count_gt) %>% pull #%>%
data_qc_sum %>% 
    group_by(TASK) %>%
    summarise(dice = mean(dice))
data_qc %>% 
    group_by(TASK) %>%
    summarise(dice = mean(dice_score, na.rm = TRUE))
# write_excel_csv(data_qc_sum, paste0(Save_dir, "/Summary_data.xslx"))

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

data_qc_sum_resh = as_tibble(melt(data_qc_sum %>% ungroup %>% select(-c(TASK, task, DICE_ID, count_gt, count_pred)), id.vars = c("strategy")))
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

