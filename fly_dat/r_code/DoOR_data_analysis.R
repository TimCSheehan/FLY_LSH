# load odor data
#install.packages("devtools")
#library(devtools)
#install_github("ropensci/DoOR.data", ref='v2.0.1')
#install_github("ropensci/DoOR.functions")
library(DoOR.data)
library(DoOR.functions)
loadData(TRUE)
library(corrplot)

# (ACP) acetophenone       -neutral [present] {-3, 10, 62}
# (BEN) benzaldehyde       -aversive [present] {5, 200, 57}
# (MBL) 3-methyl-1-butanol -neutral [present] {30, 157, 97} **here called 3-methylbutanol
# (IPA) isopentyl acetate  -attractive [present] {65, -33, 139}
# (HXA) hexyl acetate      -attractive [present] {29, -19, 44}

## Absent Hallem data set
# (MCH) 4-methylcyclohexanol -aversive [absent]
# (OCT) 3-octanol            - aversive [absent]


# NOVELTY ODOR NAMES
odor_name <- c("4-methylcyclohexanol","3-octanol","acetophenone",
              "benzaldehyde","3-methyl-butanol","isopentyl acetate",
              "hexyl acetate",'oil')
odor_name_short <-c('MCH','OCT','ACP','BEN','MBL','IPA','HXA',"OIL")
odorants = trans_id(odor_name, from = "Name")
responses <- get_normalized_responses(odorants)

# GET ALL ODORS FOR COMPARISON
all_odors <-rownames(door_response_matrix)
ODORS_ALL <- door_response_matrix[all_odors,]
dims_all <- list(colnames(ODORS_ALL),all_odors)
ODOR_ALL_MAT <- matrix(unlist(ODORS_ALL), ncol = length(all_odors), byrow = TRUE,dimnames =dims_all  )

# for export- switch to CAS name
OM_CAS <- ODOR_ALL_MAT
col_ids <- colnames(OM_CAS)
CAS_IDS <- trans_id(col_ids,from="InCHIKey",to="Name")
od <- door_default_values("odor")
res <- od[match(col_ids,od[,"InChIKey"]),"Name"]
colnames(OM_CAS) <- res
OM_NAME <- t(OM_CAS)
write.csv(OM_NAME,'DRM_NAME.csv')


ODORS <- door_response_matrix[odorants,]
dims <- list(colnames(ODORS),odor_name_short)
ODOR_MAT <- matrix(unlist(ODORS), ncol = length(odorants), byrow = TRUE,dimnames =dims  )

# get just Hallem.2006.EN data
HALLEM_dat = get_dataset('Hallem.2006.EN',na.rm=TRUE)
H_cols = colnames(HALLEM_dat) 
ORN_Hallem = H_cols[6:length(H_cols)]
ODOR_Hallem = HALLEM_dat$InChIKey
HALLEM_dat_MAT = t(HALLEM_dat[,ORN_Hallem])

# get correlation w/ full dataSet
H_ODOR_ALL_MAT = ODOR_ALL_MAT[ORN_Hallem,ODOR_Hallem]
n_have = colSums(!is.na(H_ODOR_ALL_MAT))
c_h = diag(cor(H_ODOR_ALL_MAT,HALLEM_dat_MAT,use='pairwise.complete.obs',method = 'pearson'))[n_have>15]
hist(c_h,breaks=10,xlab = 'Pearson Correlation Hallem ~ DoOR',ylab='# Odors',main='Minimum 15 entries')

# Visualize data matrix
ORN_ind  = 1:length(ODORS)
ODOR_ind = 1:length(odorants)
image(ORN_ind,ODOR_ind,ODOR_MAT,col=topo.colors(4))

# Pairwise correlations
# complete.obs | casewise row deletion
# pairwise.complete.obs | pairwise deletion
cor_odors = cor(ODOR_MAT,use = "complete",method = 'pearson')

# Visualize all
image(x=ODOR_ind,y=ODOR_ind,z=cor_odors,axes=FALSE,xlab='Odor',ylab='Odor')
axis(1, at=ODOR_ind,labels=odor_name_short, col.axis="red", las=1)
axis(2, at=rev(ODOR_ind),labels=odor_name_short, col.axis="red", las=1)
title('ORN Correlation (Pearson)')

# Visualize MCH
plot(cor_odors[1,],axes = TRUE,ylab='Correlation ~ MCB',xlab='Odor',xaxt='n')
title('Spearman Corrleatation')
axis(1, at=ODOR_ind,labels=odor_name_short, col.axis="red", las=1)


odor_supress = 1- c(.1,1,.5,.95,.65,.85,.95,1)
plot(atanh(cor_odors[1,-1]),odor_supress[-1],xlab='Correlation ~ MCH',ylab='Suprress MCH')
plot(cor_odors[1,],odor_supress,xlab='Correlation ~ MCH',ylab='Suprress MCH')
cor(cor_odors[1,-1],odor_supress[-1])
cor(cor_odors[1,],odor_supress)
cor(atanh(cor_odors[1,-1]),odor_supress[-1])


orn_plot <- function(odor_a,odor_b){
  bad = is.na(odor_a) | is.na(odor_b)
  plot(odor_a[!bad],odor_b[!bad],xlab = 'ODOR A',ylab = 'ODOR B')
}

orn_plot(ODOR_MAT[,1],ODOR_MAT[,3])


corrplot(cor_odors,method = "number",type='upper')

ODOR_df = data.frame(ODOR_MAT,row.names=colnames(ODORS))
colnames(ODOR_df) <- odor_name_short
plot(ODOR_df)

ALL_ODORS = matrix(unlist(door_response_matrix),ncol = length(door_response_matrix[[1]]),byrow=TRUE)
hist(ALL_ODORS)

x = ecdf(door_response_matrix[1,])
plot(x)


#pre_rm_hallem <- door_response_matrix
## removing hallem data
# Note, there is also Hallem.2004.EN and Hallem.2004.WT
remove_study('Hallem.2006.EN')
create_door_database()

# GET ALL ODORS NO HALLEM FOR COMPARISON
ODORS_ALL_NH <- door_response_matrix[all_odors,]
ODOR_ALL_MAT_NH <- matrix(unlist(ODORS_ALL_NH), ncol = length(all_odors), byrow = TRUE,dimnames =dims_all  )

# Correlations 
# samples per ORN
#n_samp = rowSums(!is.na(ODOR_ALL_MAT_NH))
#ORN_require = n_samp>100

# HALLEM_dat
# ODORS_ALL_MAT_NH
good_hallem <- !is.na(HALLEM_dat$Or2a)
hallem_odor_names <- HALLEM_dat[good_hallem,]$InChIKey


hallem_odors <- HALLEM_dat[good_hallem,]
hallem_orn_names <- colnames(hallem_odors)[6:29]
no_hallem_odors <- ODOR_ALL_MAT_NH[hallem_orn_names,hallem_odor_names] # try with hallem too
noy_hallem_odors <- ODOR_ALL_MAT[hallem_orn_names,hallem_odor_names] # try with hallem too
yes_hallem_odors <- hallem_odors[,hallem_orn_names]

n_samp = colSums(!is.na(noy_hallem_odors))
hist(n_samp)

cor_h_nh = cor(noy_hallem_odors,t(yes_hallem_odors),use = "pairwise.complete.obs",method = 'pearson')
diag_cor = diag(cor_h_nh)[n_samp>20]
hist(diag_cor,10)

tcor = diag(cor(noy_hallem_odors[,n_samp>20],t(yes_hallem_odors[n_samp>20,]),method='spearman'))

oo =25
orn_plot(noy_hallem_odors[,oo],yes_hallem_odors[oo,])
         
