#--------------------------------------------------------------------------
# Libraries ---------------------------------------------------------------
#--------------------------------------------------------------------------

pacman::p_load(foreign, Amelia, ComplexHeatmap, circlize, caret, smotefamily)
pacman::p_load(mice, ggplot2, doSNOW, dplyr,plotROC, ModelMetrics)
rm(list = ls(all.names = TRUE))
gc(full = TRUE)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
graphics.off()
options(stringsAsFactors = FALSE)

#--------------------------------------------------------------------------
# Read and Preprocess data ------------------------------------------------
#--------------------------------------------------------------------------
# yr1 <- read.arff('./data/1year.arff')
# yr2 <- read.arff('./data/2year.arff')
# yr3 <- read.arff('./data/3year.arff')
# yr4 <- read.arff('./data/4year.arff')
# yr5 <- read.arff('./data/5year.arff')

#--------------------------------------------------------------------------
# Read data ---------------------------------------------------------------
temp <- list.files(path = "./data", pattern = "*.arff", 
                                                  full.names = TRUE)
yr <- lapply(temp, read.arff)
# str(yr)

# res_col <- 'bankrupt'
res_col <- 'class'

chnames <- function(df){
  names(df) <- sub('class', res_col ,names(df))
  names(df) <- sub("Attr","X", names(df))
  return(df)
}
yr <- lapply(yr,chnames)

#--------------------------------------------------------------------------
# Label the data ----------------------------------------------------------
names(yr) <- paste("year", 1:length(temp), sep = "")


#--------------------------------------------------------------------------
# Data Visualization ------------------------------------------------------
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# Map missing data --------------------------------------------------------

namapfn <- function(df, pltnam = "") {
  missmap(
    df,
    col = c('grey', 'black'),
    legend = FALSE,
    # don't sort the columns
    # by # of missing NA's
    rank.order = FALSE,
    main = pltnam
  )
}
#--------------------------------------------------------------------------
# Generate maps using the fn above
namaps <- mapply(namapfn, yr, names(yr))

#--------------------------------------------------------------------------
# Calculate the pct loss if NA's are removed
yr_ <- lapply(yr, na.omit)
pctloss <- (1 - sapply(yr_, nrow) / sapply(yr, nrow))*100
pctloss

#--------------------------------------------------------------------------
# Prepare data for missing data correlation
yr_na <- lapply(sapply(sapply(yr, is.na), abs), as.data.frame)

#--------------------------------------------------------------------------
# calculate correlation only for attributes with NA values
# by checking if the sum > 0
corlis <- sapply(sapply(yr_na, function(x){x[,colSums(x) > 0]}),cor)

#--------------------------------------------------------------------------
# create a new function for parametrizing the heatmap fn ------------------
Hmapfn <- function(cmat, col_title = "") {
  Heatmap(
    cmat,
    col = colorRamp2(c(-1, 0, 1), c("green", "white", "red")),
    column_title = col_title,
    column_title_side = "top",
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_row_names = F,
    show_heatmap_legend = F,
    rect_gp = gpar(type = "none"),
    cell_fun = function(j, i, x, y, w, h, col) {
      if (i > j)
      {
        grid.rect(x, y, w, h, gp = gpar(fill = col))
        if (cmat[i, j] > 0.7) {
          grid.text(sprintf("%.1f", cmat[i, j]), x, y)
        }
      } else if (j == i)
      {
        grid.text(rownames(cmat)[i], x, y, just = 'left')
      }
    }
  )
}

#--------------------------------------------------------------------------
# Finally generate the correlation matrix graphs --------------------------
hmaps <- mapply(Hmapfn, corlis, names(corlis))
show(hmaps)

#--------------------------------------------------------------------------
# More Visualizations -----------------------------------------------------

combs <- c("X1", "X2", "X3", "X4") %>%
  combn(., 2) %>%
  t() %>%
  data.frame()
classplot <- function(x, y, z)
             {
                ggplot(yr_$year1, aes_string(x, y , color = z, fill = z)) +
                geom_point(size = 5)
             }

cmap <- mapply(classplot, combs[,1], combs[,2], res_col, SIMPLIFY = FALSE)
show(cmap)

#--------------------------------------------------------------------------
# Calculating data imbalance ----------------------------------------------
#--------------------------------------------------------------------------

bnkrupt_ct <- sapply(yr,function(x)table(x[,res_col]))
bnkrupt_ct[2,]/apply(bnkrupt_ct,2,sum)*100

bnkrupt_ct_na <- sapply(yr_,function(x)table(x[,res_col]))
bnkrupt_ct_na[2,]/apply(bnkrupt_ct_na,2,sum)*100

#--------------------------------------------------------------------------
# Missing data Imputation -------------------------------------------------
#--------------------------------------------------------------------------
#Removing NA's is not an option
#we will impute the missing data using the following two methods

#--------------------------------------------------------------------------
# 1.Mean imputation
# if(!file.exists('./rds/yr_'))
if (!file.exists('./rds/yr_avimput.rds')) {
  yr_avimput <-
    lapply(yr,mice, m = 5, maxit = 5, printFlag = FALSE,
                                              seed = 1, method = 'mean')
  saveRDS(yr_avimput, './rds/yr_avimput.rds')
}

yr_avimput <- readRDS('./rds/yr_avimput.rds')

#--------------------------------------------------------------------------
# 2.MICE - Multiple imputation using chained equations

if (!file.exists('./rds/yr_imput.rds')) {
  yr_imput <-
    lapply(yr, mice, m = 5, maxit = 5, printFlag = FALSE,
                                              seed = 1, method = 'pmm')
  saveRDS(yr_imput,'./rds/yr_imput.rds')
}

yr_imput <- readRDS('./rds/yr_imput.rds')

#--------------------------------------------------------------------------
# Choose an imputation to complete the data

yr_av <- lapply(yr_avimput, complete, 1)
yr_pmm <- lapply(yr_imput, complete, 1)

# Calculate the pct loss again if NA's are removed
yr_av <- lapply(yr_av, na.omit)
pctloss <- (1 - sapply(yr_av, nrow) / sapply(yr, nrow))*100
pctloss

yr_pmm <- lapply(yr_pmm, na.omit)
pctloss <- (1 - sapply(yr_pmm, nrow) / sapply(yr, nrow))*100
pctloss

#--------------------------------------------------------------------------
# Oversampling ------------------------------------------------------------
#--------------------------------------------------------------------------

class_col <- 65
# yr_av_OS <- sapply(yr_av, function(x)
#                           {SMOTE(x[,-class_col],x[,class_col], K = 5)[1]})

if (!file.exists('./rds/yr_pmm_OS.rds')) {
  yr_pmm_OS <- sapply(yr_pmm, function(x)
  {
    SMOTE(x[, -class_col], x[, class_col], K = 5)[1]
  })
  saveRDS(yr_pmm_OS,'./rds/yr_pmm_OS.rds')
}

yr_pmm_OS <- readRDS('./rds/yr_pmm_OS.rds')

#--------------------------------------------------------------------------
# Preprocess the imputed data ---------------------------------------------
names(yr_pmm_OS) <- names(yr_pmm)
str(yr_pmm_OS$year1)

yr_pmm_OS <- lapply(yr_pmm_OS,
                    function(x)
                      {
                      x$class <- as.factor(x$class)
                      levels(x$class) <- c("alive","bankrupt")
                      return(x)
                      }
                    )

#--------------------------------------------------------------------------
# Data partitioning -------------------------------------------------------
#--------------------------------------------------------------------------
indices <- createDataPartition(
  yr_pmm_OS$year1[,res_col],
  times = 1,
  p = 0.8,
  list = F
)

yr_pmm_OS.train <- (yr_pmm_OS$year1[indices,])
yr_pmm_OS.test <- (yr_pmm_OS$year1[-indices,])

#--------------------------------------------------------------------------
# Finally train the model -------------------------------------------------
#--------------------------------------------------------------------------

train.control <- trainControl(method = 'repeatedcv', number = 10,
                              repeats = 5,search = 'grid',
                              classProbs = TRUE,
                              preProcOptions = list(thresh = 0.95),
                              summaryFunction = twoClassSummary,
                              savePredictions = 'all')

tune.grid <- expand.grid(mtry = 2:64)

#--------------------------------------------------------------------------
# Make clusters -----------------------------------------------------------
# ML is expensive------------ ---------------------------------------------
# Lock and Load your model ------------------------------------------------

if (!file.exists('./rds/yr_pmm_OS_pca_rf.rds'))
{
  cl <- makeCluster(4, type = 'SOCK')
  registerDoSNOW(cl)

  yr_pmm_OS.rf.cv <- train(class ~ ., data = yr_pmm_OS.train,
                           method = 'rf',
                           metric = "ROC",
                           preProcess=c("center", "scale", "pca"),
                           trControl = train.control,
                           tuneGrid = tune.grid)
  stopCluster(cl)
  saveRDS(yr_pmm_OS.rf.cv,'./rds/yr_pmm_OS_pca_rf.rds')
}

yr_pmm_OS.rf.cv <- readRDS('./rds/yr_pmm_OS_pca_rf.rds')

#--------------------------------------------------------------------------
# Predict using the test data ---------------------------------------------

preds <- predict(yr_pmm_OS.rf.cv, yr_pmm_OS.test)

#--------------------------------------------------------------------------
# Analyze the results -----------------------------------------------------

confusionMatrix(preds, as.factor(yr_pmm_OS.test[,res_col]))

#--------------------------------------------------------------------------
# ROC Curve ---------------------------------------------------------------

selectedIndices <- yr_pmm_OS.rf.cv$pred$mtry ==
                                    which.max(yr_pmm_OS.rf.cv$results$ROC)

ROCdata <- yr_pmm_OS.rf.cv$pred[selectedIndices, ]
predicted <- ROCdata$bankrupt
actual <- ROCdata$obs
AUC <- round(auc(actual,predicted),4)

ggplot(ROCdata,
  aes(m = bankrupt, d = factor(obs, levels = c("alive", "bankrupt")))) +
  geom_roc() +
  coord_equal() +
  style_roc() +
  annotate("text",
           x = 0.75,
           y = 0.25,
           label = paste("AUC =", AUC))