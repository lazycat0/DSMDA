library(DOSE)

Dsim<-read.csv("E:/Data/DID.csv",header=TRUE)
Dsim

s<-doSim(Dsim["DOID"], Dsim["DOID"], measure="Wang")
s

?doSim

write.csv(s,"C:/Users/wangliugen/Desktop/microbe-disease/ddsim")

