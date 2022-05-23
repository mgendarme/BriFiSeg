library(tidyverse)
library(rstudioapi)
library(ggpubr)
library(reshape2)
library(viridis)


## Settings
# for linux full path necessary due to image export from EBImage that can't deal with relative path
RelPath = ifelse(grepl("Windows", sessionInfo()$running), '~/UNet', '/home/gendarme/Documents/UNet')
TASK = 17
nnunetpath = str_replace(RelPath, "UNet", "nnUNet")
Save_dir = paste0(nnunetpath, "/nnUNet_trained_models/nnUNet/2d/Task0", TASK, "_RWA_A549/nnUNetTrainerV2_unet_v3_noDeepSupervision_sn_adam__nnUNetPlansv2.1/WS")

# set saving directories for plots 
Save_data = paste0(Save_dir, "/data2")
Save_data_ctr = paste0(Save_dir, "/data_ctr2")
Save_data_ts = paste0(Save_dir, "/data_ts2")
Save_plot_instance = paste0(Save_dir, "/Plot_instance_ws2")
dir.create(Save_plot_instance, showWarnings = F)

# bind all data together 
list_data = c(list.files(path = Save_data, pattern = ".RData", full.names = TRUE),
              list.files(path = Save_data_ctr, pattern = ".RData", full.names = TRUE),
              list.files(path = Save_data_ts, pattern = ".RData", full.names = TRUE)
              )
             
data = tibble()
for(i in list_data){
    print(i)
    load(i)
    temp = get("single_dice_val")  %>%
        mutate(file_name = i,
               file_name = str_replace(file_name, "_ctr", ""),
               file_name = str_replace(file_name, "_ts", ""),
               file_name = sub(".*/data/", "", file_name),
               DICE_ID = sub(".*single_dice_", "", file_name),
               DICE_ID = as.numeric(sub("_ID.*", "", DICE_ID)),
               ID = sub(".*_ID--", "", file_name),
               ID = as.numeric(sub("_WELL.*", "", ID)),
               WELL = sub(".*_WELL--", "", file_name),
               WELL = sub("\\.RData.*", "", WELL)
               )
            
    temp = temp %>% 
        select(-c(file_name)) %>% 
        select(DICE_ID, ID, WELL, everything())
    data = rbind(data, temp)
}

ts_wells = sort(rep(c(paste0(LETTERS[4:9], "05"),
                      paste0(LETTERS[4:9], "06")), 4))
ts_ID = 225:272
for(i in 1:length(ts_ID)){
    print(paste0(ts_ID[i], " ", ts_wells[i]))
    temp = data %>% 
        filter(WELL == "A549" & DICE_ID == i) %>%
        mutate(WELL = str_replace(WELL, "A549", ts_wells[i]))
    data = data %>% 
        filter(!(WELL == "A549" & DICE_ID == i)) %>%
        rbind(temp)
}
data %>% select(WELL) %>% pull %>% unique
data_bu = data # helps for testing to get back original data

# Add metainformation (if necessary, more revelant for data analysis)
IdToMap = function(FileList){
    rN = c(1:16)
    cN = c(1:24)
    cID = c(1:24)
    rID = c('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P')
    Row = data.frame(rID, rN)
    ID = data.frame(cID, cN)
    ID$rID = NA
    for(i in rID){
        IDtemp = ID[1:24,]
        IDtemp$rID = as.character(i) 
        ID = rbind(ID, IDtemp)
    }
    ID = merge(ID[25:nrow(ID),], Row, by = "rID", all.x = T)
    ID$ID = paste(ID$rID, ID$cID, sep = "")
    for(i in 1:nrow(ID)){
        if(nchar(ID[i, "ID"]) == 2){
            ID[i, "ID"] = paste0(str_split(ID[i, "ID"], "")[[1]][1],
                                 0,
                                 str_split(ID[i, "ID"], "")[[1]][2])
        }
    }
    FileList = dplyr::left_join(FileList, ID, by = c("WELL" = "ID"))
}
data = IdToMap(data_bu) %>%
    select(ID, rID, cID, rN, cN, everything())
data

drug = list("Staurosporin", "Doxorubicin", "Nocodazole", "Vorinostat", "Nigericin", "Vehicle")
n = as.integer(16) # amount of wells per row
concentration = list(sort(purrr::accumulate(rep(1, n/2), ~ .x/2)),
                     sort(purrr::accumulate(rep(10, n/2), ~ .x/2)),
                     sort(purrr::accumulate(rep(1, n/2), ~ .x/2)),
                     sort(purrr::accumulate(rep(20, n/2), ~ .x/2)),
                     sort(purrr::accumulate(rep(20, n/2), ~ .x/2)),
                     0.02
                     )

data_qc = data %>%
     mutate(extra = case_when(is.na(inst_true) & extra == 0 ~ 1, TRUE ~ extra), # to rectify for extra nuclei in pred absent in gt
            Drug = case_when(
                rN %in% 3:4 & cN %in% 7:22 ~ drug[[1]],
                rN %in% 5:6 & cN %in% 7:22 ~ drug[[2]],
                rN %in% 7:8 & cN %in% 7:22 ~ drug[[3]],
                rN %in% 9:10 & cN %in% 7:22 ~ drug[[4]],
                rN %in% 11:12 & cN %in% 7:22 ~ drug[[5]],
                rN %in% 13:14 & cN %in% 7:22 ~ drug[[6]],
                rN %in% 3:14 & cN %in% 5:6 ~ drug[[6]]),
            Concentration = case_when(
                # drug 1
                rN %in% 3:4 & cN %in% 7:8 ~ concentration[[1]][1], 
                rN %in% 3:4 & cN %in% 9:10 ~ concentration[[1]][2], 
                rN %in% 3:4 & cN %in% 11:12 ~ concentration[[1]][3], 
                rN %in% 3:4 & cN %in% 13:14 ~ concentration[[1]][4], 
                rN %in% 3:4 & cN %in% 15:16 ~ concentration[[1]][5], 
                rN %in% 3:4 & cN %in% 17:18 ~ concentration[[1]][6], 
                rN %in% 3:4 & cN %in% 19:20 ~ concentration[[1]][7], 
                rN %in% 3:4 & cN %in% 21:22 ~ concentration[[1]][8], 
                # drug 2
                rN %in% 5:6 & cN %in% 7:8 ~ concentration[[2]][1], 
                rN %in% 5:6 & cN %in% 9:10 ~ concentration[[2]][2], 
                rN %in% 5:6 & cN %in% 11:12 ~ concentration[[2]][3], 
                rN %in% 5:6 & cN %in% 13:14 ~ concentration[[2]][4], 
                rN %in% 5:6 & cN %in% 15:16 ~ concentration[[2]][5], 
                rN %in% 5:6 & cN %in% 17:18 ~ concentration[[2]][6], 
                rN %in% 5:6 & cN %in% 19:20 ~ concentration[[2]][7], 
                rN %in% 5:6 & cN %in% 21:22 ~ concentration[[2]][8],
                # drug 3
                rN %in% 7:8 & cN %in% 7:8 ~ concentration[[3]][1], 
                rN %in% 7:8 & cN %in% 9:10 ~ concentration[[3]][2], 
                rN %in% 7:8 & cN %in% 11:12 ~ concentration[[3]][3], 
                rN %in% 7:8 & cN %in% 13:14 ~ concentration[[3]][4], 
                rN %in% 7:8 & cN %in% 15:16 ~ concentration[[3]][5], 
                rN %in% 7:8 & cN %in% 17:18 ~ concentration[[3]][6], 
                rN %in% 7:8 & cN %in% 19:20 ~ concentration[[3]][7], 
                rN %in% 7:8 & cN %in% 21:22 ~ concentration[[3]][8],
                # drug 4
                rN %in% 9:10 & cN %in% 7:8 ~ concentration[[4]][1], 
                rN %in% 9:10 & cN %in% 9:10 ~ concentration[[4]][2], 
                rN %in% 9:10 & cN %in% 11:12 ~ concentration[[4]][3], 
                rN %in% 9:10 & cN %in% 13:14 ~ concentration[[4]][4], 
                rN %in% 9:10 & cN %in% 15:16 ~ concentration[[4]][5], 
                rN %in% 9:10 & cN %in% 17:18 ~ concentration[[4]][6], 
                rN %in% 9:10 & cN %in% 19:20 ~ concentration[[4]][7], 
                rN %in% 9:10 & cN %in% 21:22 ~ concentration[[4]][8],
                # drug 5
                rN %in% 11:12 & cN %in% 7:8 ~ concentration[[5]][1], 
                rN %in% 11:12 & cN %in% 9:10 ~ concentration[[5]][2], 
                rN %in% 11:12 & cN %in% 11:12 ~ concentration[[5]][3], 
                rN %in% 11:12 & cN %in% 13:14 ~ concentration[[5]][4], 
                rN %in% 11:12 & cN %in% 15:16 ~ concentration[[5]][5], 
                rN %in% 11:12 & cN %in% 17:18 ~ concentration[[5]][6], 
                rN %in% 11:12 & cN %in% 19:20 ~ concentration[[5]][7], 
                rN %in% 11:12 & cN %in% 21:22 ~ concentration[[5]][8],
                # vehicle
                rN %in% 3:12 & cN %in% 5:6 ~ 0.01,
                rN %in% 13:14 & cN %in% 7:22 ~ 0.03,
                rN %in% 13:14 & cN %in% 5:6 ~ concentration[[6]][1]),
                Sample = paste0(Drug, " ", formatC(Concentration, format = "e", digits = 1), " µM")
            )

data_qc %>% filter(Drug == "Vehicle") %>% select(WELL) %>% pull %>% unique
data_qc %>% filter(Drug == "Vehicle" & Concentration == 0.01) %>% select(WELL) %>% pull %>% unique
data_qc %>% filter(Drug == "Vehicle" & Concentration == 0.02) %>% select(WELL) %>% pull %>% unique
data_qc %>% filter(Drug == "Vehicle" & Concentration == 0.03) %>% select(WELL) %>% pull %>% unique

data_qc %>%
    filter(Drug == "Vehicle" & Concentration == 0.01) %>%
    group_by(Drug, Concentration, Sample, WELL) %>%
    summarise(count_gt = sum(!is.na(inst_true))) %>%
    select(count_gt) %>% pull #%>%
data_qc %>%
    filter(Drug == "Vehicle" & Concentration == 0.02) %>%
    group_by(Drug, Concentration, Sample, WELL) %>%
    summarise(count_gt = sum(!is.na(inst_true))) %>%
    select(count_gt) %>% pull #%>%
tt = data_qc %>%
    filter(Drug == "Vehicle" & Concentration == 0.03) %>%
    group_by(Drug, Concentration, Sample, WELL) %>%
    summarise(dice = mean(dice_score, na.rm = TRUE),
              count_gt = sum(!is.na(inst_true)),
              count_pred = sum(!is.na(inst_pred)),
              undersplit = sum(undersplit, na.rm = TRUE)/n(),
              oversplit = sum(oversplit, na.rm = TRUE)/n(),
              extra = sum(extra, na.rm = TRUE)/n(),
              missed = sum(missed, na.rm = TRUE)/n()) %>% 
    mutate(count = count_pred / count_gt) %>%
    select(Drug, Concentration, dice, count, everything())
    # select(count_gt) %>% pull #%>%

data_qc_sum = data_qc %>%
    filter(!is.na(Drug)) %>%
    filter(!(Drug == "Vehicle" & Concentration == 0.01)) %>%
    filter(!(cN %in% 13:22 & rN %in% 13:14)) %>%
    group_by(Drug, Concentration, Sample, WELL) %>%
    summarise(dice = mean(dice_score, na.rm = TRUE),
              count_gt = sum(!is.na(inst_true)),
              count_pred = sum(!is.na(inst_pred)),
              undersplit = sum(undersplit, na.rm = TRUE)/n(),
              oversplit = sum(oversplit, na.rm = TRUE)/n(),
              extra = sum(extra, na.rm = TRUE)/n(),
              missed = sum(missed, na.rm = TRUE)/n()
              ) %>%
    mutate(count = count_pred / count_gt) %>%
    select(Drug, Concentration, dice, count, everything())

vehicle_cell_count_gt = data_qc_sum %>%
    ungroup() %>%
    filter(Drug == "Vehicle") %>%
    select(count_gt) %>%
    pull %>%
    mean()
vehicle_cell_count_pred = data_qc_sum %>%
    ungroup() %>%
    filter(Drug == "Vehicle") %>%
    select(count_pred) %>%
    pull %>%
    mean()

data_qc_sum %>% filter(Drug == "Vehicle")
mean(data_qc$dice_score, na.rm = TRUE)

data_qc_sum = data_qc_sum %>%
    mutate(count_gt_rel = count_gt / vehicle_cell_count_gt,
           count_pred_rel = count_pred / vehicle_cell_count_pred)

# write_excel_csv(data_qc_sum, paste0(Save_dir, "/Summary_data.xslx"))

for(plt in c("dice", "count", "count_gt", "count_pred", "count_gt_rel", "count_pred_rel",
             "missed", "extra", "oversplit", "undersplit")){

    plot = data_qc_sum %>% 
        ungroup() %>%
        mutate(Concentration = round(Concentration, 2)) %>%
        ggerrorplot(x = "Concentration",
                    y = plt,
                    desc_stat = "mean_sd",
                    facet.by = "Drug") +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

    ggsave(paste0(Save_plot_instance, "/", plt, ".png"), plot, width = 12, height = 6)

}

data_qc_sum_resh = as_tibble(melt(data_qc_sum, id.vars = c("Drug", "Concentration", "Sample", "WELL")))
data_qc_sum_resh %>% select(variable) %>% pull %>% unique


plot_gr1 = data_qc_sum_resh %>% 
        filter(variable %in% c("dice",
                               "count_gt_rel",
                               "count_pred_rel"#,
                            #    "count"
                               )) %>%
        mutate(variable = case_when(variable == "dice" ~ "Dice",
                         variable == "count_gt_rel" ~ "Rel. true cell count",
                         variable == "count_pred_rel" ~ "Rel. pred. cell count")) %>%
        mutate(Concentration = round(Concentration, 2)) %>%
        mutate(Drug = factor(Drug, levels = c("Vehicle", "Staurosporin", "Doxorubicin", "Nocodazole", "Vorinostat", "Nigericin"))) %>% 
        ggerrorplot(x = "Concentration",
                    y = "value",
                    color = "variable",
                    desc_stat = "mean_sd",
                    facet.by = "Drug",
                    legend.title = "Measure") + #,
                    # palette = c("#00AFBB", "#E7B800", "#FC4E07")) +
        theme_classic2() +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
        scale_color_viridis(discrete = TRUE) +
        scale_y_continuous(breaks = get_breaks(n = 11)) +
        theme(
        panel.background = element_rect(fill = NA),
        panel.grid.major = element_line(colour = "grey90"),
        panel.ontop = FALSE
        ) + 
        theme(legend.position="top")
        # scale_color_grey(start = 0.7, end = 0.3) 
ggsave(paste0(Save_plot_instance, "/", "pltgrp1", ".png"), plot_gr1, width = 8.5, height = 6)

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
        mutate(Concentration = round(Concentration, 2)) %>%
        mutate(Drug = factor(Drug, levels = c("Vehicle", "Staurosporin", "Doxorubicin", "Nocodazole", "Vorinostat", "Nigericin"))) %>% 
        ggerrorplot(x = "Concentration",
                    y = "value",
                    color = "variable",
                    desc_stat = "mean_sd",
                    facet.by = "Drug",
                    legend.title = "Measure") +#,
        # scale_y_break(c(0.2, 0.5)) +
                    # palette = c("#00AFBB", "#E7B800", "#FC4E07")) +
        theme_classic2() +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
        scale_color_viridis(discrete = TRUE) +         
        scale_y_continuous(breaks = get_breaks(n = 10)) +
        theme(
        panel.background = element_rect(fill = NA),
        panel.grid.major = element_line(colour = "grey90"),
        panel.ontop = FALSE
        ) + 
        theme(legend.position="top")
        # scale_color_grey(start = 0.7, end = 0.3) 
ggsave(paste0(Save_plot_instance, "/", "pltgrp2", ".png"), plot_gr2, width = 8.5, height = 6)


# library(ggbreak)
plot_gr3 = data_qc_sum_resh %>% 
        filter(variable %in% c("extra",
                               "missed",
                               "undersplit",
                               "oversplit"
                               )) %>%
        mutate(variable = case_when(variable == "extra" ~ "Extra",
                         variable == "missed" ~ "Missed",
                         variable == "undersplit" ~ "Undersplit",
                         variable == "oversplit" ~ "Oversplit")) %>%
        mutate(Concentration = round(Concentration, 2)) %>%
        mutate(Drug = factor(Drug, levels = c("Vehicle", "Staurosporin", "Doxorubicin", "Nocodazole", "Vorinostat", "Nigericin"))) %>% 
        ggerrorplot(x = "Concentration",
                    y = "value",
                    color = "variable",
                    desc_stat = "mean_sd",
                    facet.by = "Drug",
                    legend.title = "Measure") +#,
        # scale_y_break(c(0.2, 0.5)) +
                    # palette = c("#00AFBB", "#E7B800", "#FC4E07")) +
        theme_classic2() +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
        scale_color_viridis(discrete = TRUE) +         
        scale_y_continuous(breaks = get_breaks(n = 10)) +
        theme(
        panel.background = element_rect(fill = NA),
        panel.grid.major = element_line(colour = "grey90"),
        panel.ontop = FALSE
        ) +
        ylim(c(0, 0.3)) +
        theme(legend.position="top")
        # scale_color_grey(start = 0.7, end = 0.3) 
ggsave(paste0(Save_plot_instance, "/", "pltgrp3", ".png"), plot_gr3, width = 8.5, height = 6)
