data = read.csv('survey_data_simplified.csv')
library(psych)
library(GPArotation)
loadings_fa <- fa(data, nfactors=3, rotate='oblimin')
loadings_factanal = factanal(data, factors=3, rotation="none")
loadings_factanal_oblimin = GPFoblq(loadings(loadings_factanal), method="oblimin", normalize=FALSE)

write.csv(loadings(loadings_factanal_oblimin), "factanal_oblimin.csv", row.names=FALSE)
write.csv(loadings(loadings_fa), "fa_oblimin.csv", row.names=FALSE)