## These scripts apply a supervised learning approach (kNN classifier) to predict drug sensitivity for a panel of cell lines based on gene expression profiles.
## These data were taken from the 2012 NCI-DREAM Drug Sensitivity Prediction Challenge.  The cell lines were derived from a set of breast tumors or normal tissue. Each of the cell lines was exposed independently to five drugs, and the GI50 was measured. GI50 represents the rug concentration at which growth is inhibited by 50%. The GI50 values have been – log10 transformed so higher values reflect greater sensitivity. For simplicity, in this assignment we have binarized the drug sensitivity values such that “1” corresponds to sensitive and “0” corresponds to resistant.

# This is a kNN classifier that predicts which cell lines are sensitive to each drug
