require(tidyverse)

# load(paste0(
#   "/home/gendarme/Documents/UNet/BF_Data/A549/Prediction/ImgBF512_2Class--5/fpn--seresnext101--Epochs_200--Minibatches_50--FOLD_1/Plot/",
#   "HTR_data_ImgBF512_2Class_fpn--seresnext101--Epochs_200--Minibatches_50--FOLD_1.rdata"
# ))

# Value used to transform the data
coeff <- 1

# A few constants
# temperatureColor <- "#69b3a2"
# priceColor <- rgb(0.2, 0.6, 0.9, 1)
loss_col = "blue"
val_loss_col = "red"
val_dice_col = "green"

data = tibble(epochs = 1:history_data$params$epochs,
              loss = history_data$metrics$loss,
              val_loss = history_data$metrics$val_loss,
              dice = 1-history_data$metrics$dice,
              val_dice = 1-history_data$metrics$val_dice,
              lr = history_data$metrics$lr)

round(history_data$metrics$val_loss, 3) * 100

# history_data$metrics$val_dice[175:200]
# # Function factory for secondary axis transforms
# train_sec <- function(primary, secondary, na.rm = TRUE) {
#   # Thanks Henry Holm for including the na.rm argument!
#   from <- range(secondary, na.rm = na.rm)
#   to   <- range(primary, na.rm = na.rm)
#   # Forward transform for the data
#   forward <- function(x) {
#     rescale(x, from = from, to = to)
#   }
#   # Reverse transform for the secondary axis
#   reverse <- function(x) {
#     rescale(x, from = to, to = from)
#   }
#   list(fwd = forward, rev = reverse)
# }


## https://www.biostars.org/p/412685/

# range have to match => same difference
if(CLASS == 1){
  ylim_loss = c(0.0, 0.15) #range(data$loss)#c(0.0, 0.2)
  ylim_dice = c(0.80, 0.95) #range(data$dice)#c(0.8, 1.0)
} else if(CLASS == 2){
  ylim_loss = c(0.0, 0.25) #range(data$loss)#c(0.0, 0.2)
  ylim_dice = c(0.60, 0.85) #range(data$dice)#c(0.8, 1.0)
}

SCALES = 0.01
scaled_breaks = seq(ylim_dice[2] - ylim_loss[2], ylim_dice[2], by = SCALES)

b <- diff(ylim_loss)/diff(ylim_dice)
a <- b*(ylim_loss[1] - ylim_dice[1])

htr_plot = ggplot(data, aes(x=epochs)) +
  
  geom_line( aes(y = loss, color='loss'), size=0.25) +
  geom_line( aes(y = val_loss, color='val_loss'), size=0.25) +
  # geom_line( aes(y = a + dice*b, color='dice'), size=0.25) + 
  geom_line( aes(y = a + val_dice*b, color='val_dice'), size=0.25) + 
  
  scale_x_continuous(
    limits = c(0, max(data$epochs)), breaks = seq(0, max(data$epochs), 10),
    minor_breaks = NULL
  ) +
  scale_y_continuous(
    
    # Features of the first axis
    name = "loss",
    limits = c(0, ylim_loss[2]), breaks = seq(0, ylim_loss[2], SCALES),
    minor_breaks = NULL,
    # Add a second axis and specify its features
    # sec.axis = sec_axis(~.*coeff, name="loss")
    sec.axis = sec_axis(~(. - a) * b, name="dice", breaks = scaled_breaks)
  ) + 
  
  theme_light()+
  # theme_ipsum() +
  
  theme(
    axis.title.y = element_text(size=13),
    axis.title.y.right = element_text(size=13),
    legend.position="top" #c(0.9, 0.9)
  ) + #+ ggtitle("Temperature down, price up")
  scale_color_identity(guide = "legend") +
  # scale_fill_identity(name = 'the fill', guide = 'legend',labels = c('m1')) +
  scale_colour_manual(name = '', 
                      values =c('loss' = 'blue',
                                'val_loss'='red',
                                # 'dice' = 'black',
                                'val_dice'='green'
                                ),
                      breaks = c('loss','val_loss','val_dice'))#,
                      #labels = c('train_loss','val_dice', "val_loss"))
# htr_plot
ggsave(filename = paste0("HTR4_", loop_id, "_loss_valloss_valdice.png"), plot = htr_plot,
       width = 6*1.235, height = 6, dpi = 1000, path = paste0(Save_plot_semantic, "/"))
# ggsave(filename = paste0("HTR4_", "loss_valloss_valdice.png"), plot = htr_plot,
#        width = 6*1.235, height = 6, dpi = 1000, path = paste0("/home/gendarme/Desktop"))

# library(ggpubr)
# library(cowplot)
# # Demo data
# set.seed(1234)
# wdata = data.frame(
#   sex = factor(rep(c("F", "M"), each=200)),
#   weight = c(rnorm(200, 55), rnorm(200, 58)))
# 
# # Plot
# phist <- gghistogram(
#   wdata, x = "weight", 
#   add = "mean", rug = TRUE,
#   fill = "sex", palette = c("#00AFBB", "#E7B800")
# )
# #> Warning: Using `bins = 30` by default. Pick better value with the argument
# #> `bins`.
# 
# # Density plot
# pdensity <- ggdensity(
#   wdata, x = "weight", 
#   color= "sex", palette = c("#00AFBB", "#E7B800"),
#   alpha = 0, xlab = 
# ) +
#   theme_half_open(11, rel_small = 1) +
#   scale_y_continuous(
#     expand = expansion(mult = c(0, 0.05)),
#     position = "right"
#   )  +
#   theme(
#     axis.line.x = element_blank(),
#     axis.text.x = element_blank(),
#     axis.title.x = element_blank(),
#     axis.ticks = element_blank(),
#     axis.ticks.length = grid::unit(0, "pt")
#   )
# 
# # Aligning histogram and density plots
# aligned_plots <- align_plots(phist, pdensity, align="hv", axis="tblr")
# ggdraw(aligned_plots[[1]]) + draw_plot(aligned_plots[[2]])



## plot the pixel intensity distribution to check imagenet preprocessing 
# img_data = tibble(
#   r = array(input$X[[1]][,,1], dim = dim(input$X[[1]][,,1])[1]*dim(input$X[[1]][,,1])[2]),
#   g = array(input$X[[1]][,,2], dim = dim(input$X[[1]][,,2])[1]*dim(input$X[[1]][,,2])[2]),
#   b = array(input$X[[1]][,,3], dim = dim(input$X[[1]][,,3])[1]*dim(input$X[[1]][,,3])[2])
#   )

# img_data_reshape = tibble()

# for(i in 1:ncol(img_data)){
#   temp = tibble(Value = select(img_data, i) %>% pull,
#                 Channel = as.character(i))
#   img_data_reshape = rbind(img_data_reshape, temp)
# }

# img_hist_plot = 
#   ggplot(img_data_reshape, aes(x = Value, fill = Channel)) + 
#     geom_histogram(position = "identity", alpha = 0.2, bins = 50)

# img_hist_plot

# ggsave(filename = paste0("img_hist_imagenet_preprocessing.png"), plot = img_hist_plot,
#        width = 6, height = 6, dpi = 1000, path = Save_plot)
