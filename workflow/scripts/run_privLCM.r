args = commandArgs(trailingOnly = TRUE)
epsilon = as.numeric(args[[1]])
repeat_ind = as.integer(as.numeric(args[[2]]))
output_dir = args[[3]]
# repeat_ind = 10
# epsilon = 1.0

set.seed(repeat_ind + 1000 * epsilon)
N=2000
P=3

##Simulating data
prob_var = c(0.5, 0.5)
var1 = sample(1:2, N, replace = TRUE, prob = c(prob_var[1], 1-prob_var[1]))
var2 = sample(1:2, N, replace = TRUE, prob = c(prob_var[2], 1-prob_var[2]))
# var3 = sample(1:2, N, replace = TRUE, prob = c(0.5, 0.99))
theta = 1 / (1 + exp(-(var1 - 1)))
var3 = c()
for(i in 1:N){
  var3 = c(var3, sample(1:2, 1, replace=TRUE, prob=c(1 - theta[i], theta[i])))
}

##Creating the data frame
df = data.frame("Var1" = var1,
                "Var2" = var2,
                "Var3" = var3)

head(df)
colSums(df - 1)

freq = c()
P = 3
comb.mat = matrix(ncol = P)

for(i in 1:(P-1)){
  for(j in (i+1):P){
    counts = plyr::count(df[,c(i,j)])
    freq = c(freq, counts$freq)
    tmp.mat = matrix(NA, nrow = 4, ncol = P)
    tmp.mat[,i] = counts[,1]
    tmp.mat[,j] = counts[,2]
    comb.mat = rbind(comb.mat, tmp.mat)
  }
}

comb.mat = comb.mat[-1,]

library(privLCM)
library(data.table)
library(Rfast)
library(parallel)

cl = makeCluster(detectCores()-1)
# clusterExport(cl, varlist = c("twoWay.probs.vectorized", "row.prob.vec","getProbs", "data.table", "rowprods", "P"))

samps = mcmc.sampler(freq, comb.mat, eps = epsilon, P = P, G = 10, nsamples = 500, samp.size = N, cl = cl, .PiTuningParam = 0.075, .PsiTuningParam =75, .calculateFullTabProbs = TRUE)
# stopCluster(cl)

n_syn_datasets = 100
n_syn_dataset = N
x_values = matrix(c(0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1), ncol=3, nrow=8, byrow=TRUE)
inds = sample(2:dim(samps$full_probs)[1], n_syn_datasets, replace=TRUE)
probs = samps$full_probs[inds, ]
for(i in 1:n_syn_datasets){
  inds = sample(1:8, n_syn_dataset, replace=TRUE, prob=probs[i, ])
  syn_data = x_values[inds, ]
  output_file <- paste("privLCM_", repeat_ind, "_", epsilon, "_", i - 1, ".csv", sep = "")
  write.csv(syn_data, paste(output_dir, output_file, sep=""), row.names=FALSE)
#   write.csv(syn_data, output_file, row.names=FALSE)
}
