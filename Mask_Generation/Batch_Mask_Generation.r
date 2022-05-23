for(i in c(
    "BATCH_A549",   # DONE
    "BATCH_HELA",   # DONE
    "BATCH_MCF7",   # DONE
    "BATCH_RPE1"    # DONE
    # "BATCH_THP1"  # PROBLEMATIC
    )){
    CELL = i
    source("~/BFSeg/Scripts/Mask_Generation/Mask_Nuc_BF_Batch_RPE1_MCF7_HELA_v3.R")
}
