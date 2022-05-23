for(i in c(
    "BATCH_A549",
    "BATCH_HELA",
    "BATCH_MCF7",
    "BATCH_RPE1"
    # "BATCH_THP1"
    )){
    CELL = i
    source("~/BFSeg/Scripts/Mask_Generation/Mask_Nuc_BF_Batch_RPE1_MCF7_HELA_v3.R")
}
