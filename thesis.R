## loading packages
library(R.matlab) # opening data
library(dplyr) 
library(ggplot2)
library(caret)
library(e1071) # SVM
library(kernlab)
library(factoextra) # PCA visualization
library(ROSE) # upsampling
library(hydroGOF) # RMSE
library(UBL) # upsampling for regression

## set working directory 
setwd("~/Master data science/Thesis/Depression_Severity_Interviews_-_DSI_Database/data")

data1 <- readMat("023165_07.mat")
data2 <- readMat("023165_21.mat")

## all interviews in one dataframe
interviews <- list.files(pattern = ".mat")
all_data <- sapply(lapply(interviews, readMat), as.data.frame, simplify = FALSE) %>%
  bind_rows(.id = NULL)
str(all_data)

## select variables
data <- select(all_data, "subject.id", "visit.week", "hrsd.score", contains("face.dae"), 
               contains("head.pose"), contains("head.dae"), contains("vocal"))

## mean over frames
mean_frames <- aggregate(data[,-1:-2], data[,1:2], mean)

## max over frames
max_frames <- aggregate(data[,-1:-2], data[,1:2], max)

## dividing in classes
table(mean_frames["hrsd.score"])
freq <- mean_frames %>%
  group_by(hrsd.score) %>%
  summarise(counts = n())
ggplot(freq, aes(x = hrsd.score, y = counts)) +
  geom_bar(stat = "identity") + 
  xlab("hrsd") +
  ylab("frequency")
mean(mean_frames$hrsd.score)
mean(max_frames$hrsd.score)
median(max_frames$hrsd.score)

three_classes <- mean_frames %>%
  mutate(class = case_when(.$hrsd.score <= 7 ~ "non-depressed",
                           .$hrsd.score >= 8 & .$hrsd.score < 15 ~ "mild depressed",
                           .$hrsd.score >= 15 ~ "moderate to severe depressed"))

binary <- mean_frames %>%
  mutate(class = case_when(.$hrsd.score <= 7 ~ "non-depressed",
                           .$hrsd.score > 7 ~ "depressed"))

three_classes_max <- max_frames %>%
  mutate(class = case_when(.$hrsd.score <= 7 ~ "non-depressed",
                           .$hrsd.score >= 8 & .$hrsd.score < 15 ~ "mild depressed",
                           .$hrsd.score >= 15 ~ "moderate to severe depressed"))

binary_max <- max_frames %>%
  mutate(class = case_when(.$hrsd.score <= 7 ~ "non-depressed",
                           .$hrsd.score > 7 ~ "depressed"))

table(three_classes$class)
table(binary$class)

## dataset for analysis
# binary, all features
binary_class <- select(binary, "class", contains("face.dae"), contains("head.pose"), 
                       contains("head.dae"), contains("vocal"))
binary_class <- binary_class[-56,]

binary_class_max <- select(binary_max, "class", contains("face.dae"), contains("head.pose"), 
                       contains("head.dae"), contains("vocal"))
binary_class_max <- binary_class_max[-56,]

# multiclass, all features
multiclass <- select(three_classes, "class", contains("face.dae"), contains("head.pose"),
                     contains("head.dae"), contains("vocal"))
multiclass <- multiclass[-56,]

multiclass_max <- select(three_classes_max, "class", contains("face.dae"), contains("head.pose"),
                     contains("head.dae"), contains("vocal"))
multiclass_max <- multiclass_max[-56,]

# regression, all features
regression <- select(mean_frames, "hrsd.score", contains("face.dae"), 
                     contains("head.pose"), contains("head.dae"), contains("vocal"))
regression <- regression[-56,]

regressionmax <- select(max_frames, "hrsd.score", contains("face.dae"),
                        contains("head.pose"), contains("head.dae"), contains("vocal"))
regressionmax <- regressionmax[-56,]

regressionmaxup <- regressionmax %>%
  mutate(group = case_when(.$hrsd.score <= 7 ~ "non-depressed",
                           .$hrsd.score >= 8 & .$hrsd.score < 15 ~ "mild depressed",
                           .$hrsd.score >= 15 ~ "moderate to severe depressed"))

# binary, features separately
binary_headpose <- select(binary_class_max, "class", contains("head.pose"))
binary_headmove <- select(binary_class_max, "class", contains("head.dae"))
binary_facemove <- select(binary_class_max, "class", contains("face.dae"))
binary_vocal <- select(binary_class_max, "class", contains("vocal.feats"))

# 3-level, features separately
multi_headpose <- select(multiclass_max, "class", contains("head.pose"))
multi_headmove <- select(multiclass_max, "class", contains("head.dae"))
multi_facemove <- select(multiclass_max, "class", contains("face.dae"))
multi_vocal <- select(multiclass_max, "class", contains("vocal.feats"))

########################################################### starting analyses

####### first try analysis, with 0.8/0.2 training/test set for binary and multiclass
## partitioning data for binary classification
set.seed(123)
training_index = createDataPartition(y = binary_class$class, 
                                     p = 0.80, list = FALSE)
training_set = binary_class[training_index, ]
test_set = binary_class[-training_index, ]

## first SVM model for binary classification
svm_model <- svm(formula = class ~ ., data = training_set, type = "C-classification",
                 kernel = "radial")
summary(svm_model)

pred <- predict(svm_model, training_set)
matrix <- confusionMatrix(pred, as.factor(training_set$class))
matrix$byClass

tuned_par <- tune(svm, as.factor(class) ~ ., data = training_set, type = "C-classification",
                  kernel = "radial", ranges = list(cost = 10^seq(-3,3), gamma = 1^(-1:1)))
summary(tuned_par)

svm_model_after_tune <- svm(formula = as.factor(class) ~ ., data = test_set, type = "C-classification",
                            kernel = "radial", cost = 0.001, gamma = 1)
summary(svm_model_after_tune)

pred_after_tune <- predict(svm_model_after_tune, test_set)
matrix_after_tune <- confusionMatrix(pred_after_tune, as.factor(test_set$class))
matrix_after_tune$byClass

## partitioning data for multiclass classification
set.seed(123)
training_index = createDataPartition(y = multiclass$class, 
                                     p = 0.80, list = FALSE)
training_set = multiclass[training_index, ]
test_set = multiclass[-training_index, ]

## first SVM model for three class classification
svm_model2 <- svm(formula = class ~ ., data = training_set, type = "C-classification", 
                  kernel = "radial")
summary(svm_model2)

pred2 <- predict(svm_model2, training_set)
matrix2 <- confusionMatrix(pred2, as.factor(training_set$class))
matrix2$byClass

tuned_par2 <- tune(svm, as.factor(class) ~ ., data = training_set, type = "C-classification",
                                kernel = "radial", ranges = list(cost = 10^seq(-3,3), gamma = 1^(-1:1)))
summary(tuned_par2)

svm_model_after_tune2 <- svm(formula = as.factor(class) ~ ., data = test_set, type = "C-classification",
                            kernel = "radial", cost = 0.001, gamma = 1)
summary(svm_model_after_tune2)

pred_after_tune2 <- predict(svm_model_after_tune2, test_set)
matrix_after_tune2 <- confusionMatrix(pred_after_tune2, as.factor(test_set$class))
matrix_after_tune2$byClass

##################### first try with training multimodals separately
##### binary classification for face movement
## dividing dataset
set.seed(123)
training_index = createDataPartition(y = binary_facemove$class, 
                                     p = 0.75, list = FALSE)
training_set = binary_class[training_index, ]
test_set = binary_class[-training_index, ]

svm_model_binaryface <- svm(formula = class ~ ., data = training_set, type = "C-classification", 
                  kernel = "radial")
summary(svm_model_binaryface)

pred_binaryface <- predict(svm_model_binaryface, training_set)
matrix_binaryface <- confusionMatrix(pred_binaryface, as.factor(training_set$class))
matrix$byClass

tuned_par_binaryface <- tune(svm, as.factor(class) ~ ., data = training_set, 
                             type = "C-classification", kernel = "radial", 
                             ranges = list(cost = 10^seq(-3,3), gamma = 1^(-1:1)))
summary(tuned_par_binaryface)

svm_model_after_tune_binaryface <- svm(formula = as.factor(class) ~ ., data = test_set, 
                                       type = "C-classification", kernel = "radial", 
                                       cost = 0.001, gamma = 1)
summary(svm_model_after_tune2)

pred_after_tune_binaryface <- predict(svm_model_after_tune_binaryface, test_set)
matrix_after_tune_binaryface <- confusionMatrix(pred_after_tune_binaryface, 
                                                as.factor(test_set$class))
matrix$byClass

######## Multiclass classification with caret
set.seed(123)
training_index = createDataPartition(y = multiclass$class, 
                                     p = 0.75, list = FALSE)
training_set = multiclass[training_index, ]
test_set = multiclass[-training_index, ]

# fit the model
train_control <- trainControl (method = "cv",
                               number = 10,
                               summaryFunction = multiClassSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1))
svm_multi <- train(class ~ ., data = training_set,
                   method = "svmRadial",
                   trControl = train_control,
                   preProcess = c("center", "scale"),
                   tuneGrid = grid,
                   tuneLength = 10,
                   metric = "Accuracy")
plot(svm_multi)
pred_trainmulti <- predict(svm_multi, training_set)
matrix_trainmulti <- confusionMatrix(pred_trainmulti, as.factor(training_set$class))
#predict
pred_multi <- predict(svm_multi, test_set)
matrix_multi <- confusionMatrix(pred_multi, as.factor(test_set$class))
matrix_multi$byClass

########## binary classification with caret
set.seed(123)
training_index = createDataPartition(y = binary_class$class, 
                                     p = 0.75, list = FALSE)
training_set = binary_class[training_index, ]
test_set = binary_class[-training_index, ]

# fit the model
train_control <- trainControl (method = "cv",
                               number = 10,
                               summaryFunction = prSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1))
svm_multi2 <- train(class ~ ., data = training_set,
                   method = "svmRadial",
                   trControl = train_control,
                   preProcess = c("center", "scale"),
                   tuneGrid = grid,
                   tuneLength = 10,
                   metric = "F")
plot(svm_multi2)
pred_trainmulti2 <- predict(svm_multi2, training_set)
matrix_trainmulti2 <- confusionMatrix(pred_trainmulti2, as.factor(training_set$class))
#predict
pred_multi2 <- predict(svm_multi2, test_set)
matrix_multi2 <- confusionMatrix(pred_multi2, as.factor(test_set$class))
matrix_multi2$byClass

######## regression with caret
set.seed(123)
training_index = createDataPartition(y = regression$hrsd.score, 
                                     p = 0.80, list = FALSE)
training_set = regression[training_index, ]
test_set = regression[-training_index, ]

# fit the model
train_control <- trainControl (method = "cv",
                               number = 10,
                               summaryFunction = defaultSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1))
svm_multi3 <- train(hrsd.score ~ ., data = training_set,
                    method = "svmRadial",
                    trControl = train_control,
                    preProcess = c("center", "scale"),
                    tuneGrid = grid,
                    tuneLength = 10,
                    metric = "Accuracy")
plot(svm_multi3)
pred_trainmulti3 <- predict(svm_multi3, training_set)
matrix_trainmulti3 <- confusionMatrix(pred_trainmulti3, 
                                      as.factor(training_set$hrsd.score))
#predict
pred_multi3 <- predict(svm_multi3, test_set)
matrix_multi3 <- confusionMatrix(pred_multi3, as.factor(test_set$hrsd.score))
matrix_multi3$byClass

######## binary classification for face movement with caret
# dividing dataset into training and test
set.seed(123)
training_index = createDataPartition(y = binary_facemove$class, 
                                     p = 0.80, list = FALSE)
training_set = binary_facemove[training_index, ]
test_set = binary_facemove[-training_index, ]

# fit the model
train_control <- trainControl(method = "cv", 
                              number=10, 
                              summaryFunction = prSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1))
svm_binaryfacemove <- train(class ~ ., data = training_set, 
                            method = "svmRadial", 
                            trControl = train_control, 
                            preProcess = c("center", "scale"),
                            tuneGrid = grid,
                            tuneLength = 10,
                            metric = "F")

# predict
pred_binaryfacemove <- predict(svm_binaryfacemove, test_set)
matrix_binaryfacemove <- confusionMatrix(pred_binaryfacemove, as.factor(test_set$class))
matrix_binaryfacemove$byClass

######## binary classification for head movement with caret
# dividing dataset into training and test
set.seed(111)
training_index = createDataPartition(y = binary_headmove$class, 
                                     p = 0.80, list = FALSE)
training_set = binary_headmove[training_index, ]
test_set = binary_headmove[-training_index, ]

# fit model
train_control <- trainControl(method = "repeatedcv", 
                              number=10, 
                              repeats=3,
                              summaryFunction = prSummary)
grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
svm_binaryheadmove <- train(class ~ ., data = training_set, 
                            method = "svmLinear", 
                            trControl = train_control, 
                            preProcess = c("center", "scale"),
                            tuneGrid = grid,
                            tuneLength = 10,
                            metric = "F")
pred_binaryheadmovetrain <- predict(svm_binaryheadmove, training_set)
matrix_binaryheadmovetrain <- confusionMatrix(pred_binaryheadmovetrain, as.factor(training_set$class))
plot(svm_binaryheadmove)
svm_binaryheadmove$bestTune
# pred
pred_binaryheadmove <- predict(svm_binaryheadmove, test_set)
matrix_binaryheadmove <- confusionMatrix(pred_binaryheadmove, as.factor(test_set$class))

########################################################################################
### PCA binary classification
label <- binary_class$class
binary_class_pca <- binary_class[-1]
pca <- princomp(na.omit(binary_class_pca), cor = TRUE, scores = TRUE)
fviz_eig(pca)
plot(pca)
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
             )
pca_data <- predict(pca, binary_class)
pcadata <- as.data.frame(pca_data)
pcadata <- cbind(label, pcadata)
pcadata1 <- select(pcadata, 1:8)

set.seed(123)
training_index = createDataPartition(y = pcadata1$label, 
                                     p = 0.75, list = FALSE)
training_set = pcadata1[training_index, ]
test_set = pcadata1[-training_index, ]

train_control <- trainControl (method = "cv",
                               number = 10,
                               summaryFunction = multiClassSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1))
svm_pca_binary <- train(label ~., data = training_set,
                        method = "svmRadial",
                        trControl = train_control,
                        tuneGrid = grid,
                        tuneLength = 10,
                        metric = "Accuracy")
plot(svm_pca_binary)
pred_train_pca_binary <- predict(svm_pca_binary, training_set)
matrix_train_pca_binary <- confusionMatrix(pred_train_pca_binary, as.factor(training_set$label))
#predict
pred_test_pca_binary <- predict(svm_pca_binary, test_set)
matrix_test_pca_binary <- confusionMatrix(pred_test_pca_binary, as.factor(test_set$label))
matrix_test_pca_binary$byClass

### PCA multiclass classification
label <- multiclass$class
multi_class_pca <- multiclass[-1]
pca <- princomp(na.omit(multi_class_pca), cor = TRUE, scores = TRUE)
fviz_eig(pca)
plot(pca)
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
pca_data <- predict(pca, multiclass)
pcadata <- as.data.frame(pca_data)
pcadata <- cbind(label, pcadata)
pcadata1 <- select(pcadata, 1:7)

set.seed(123)
training_index = createDataPartition(y = pcadata1$label, 
                                     p = 0.75, list = FALSE)
training_set = pcadata1[training_index, ]
test_set = pcadata1[-training_index, ]

train_control <- trainControl (method = "cv",
                               number = 10,
                               summaryFunction = multiClassSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1))
svm_pca_multi <- train(label ~., data = training_set,
                        method = "svmRadial",
                        trControl = train_control,
                        tuneGrid = grid,
                        tuneLength = 10,
                        metric = "Accuracy")
plot(svm_pca_multi)
pred_train_pca_multi <- predict(svm_pca_multi, training_set)
matrix_train_pca_multi <- confusionMatrix(pred_train_pca_multi, as.factor(training_set$label))
#predict
pred_test_pca_multi <- predict(svm_pca_multi, test_set)
matrix_test_pca_multi <- confusionMatrix(pred_test_pca_multi, as.factor(test_set$label))
matrix_test_pca_multi$byClass

### PCA regression (five step)
label <- five_step$class
five_step_pca <- five_step[-31]
pca <- princomp(na.omit(five_step_pca), cor = TRUE, scores = TRUE)
fviz_eig(pca)
plot(pca)
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
pca_data <- predict(pca, five_step)
pcadata <- as.data.frame(pca_data)
pcadata <- cbind(label, pcadata)
pcadata1 <- select(pcadata, 1:15)

set.seed(123)
training_index = createDataPartition(y = pcadata1$label, 
                                     p = 0.75, list = FALSE)
training_set = pcadata1[training_index, ]
test_set = pcadata1[-training_index, ]

train_control <- trainControl (method = "cv",
                               number = 10,
                               summaryFunction = multiClassSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1))
svm_pca_five_step <- train(label ~., data = training_set,
                       method = "svmRadial",
                       trControl = train_control,
                       tuneGrid = grid,
                       tuneLength = 10,
                       metric = "Accuracy")
plot(svm_pca_five_step)
pred_train_pca_five_step <- predict(svm_pca_five_step, training_set)
matrix_train_pca_five_step <- confusionMatrix(pred_train_pca_five_step, 
                                              as.factor(training_set$label))
#predict
pred_test_pca_five_step <- predict(svm_pca_five_step, test_set)
matrix_test_pca_five_step <- confusionMatrix(pred_test_pca_five_step, as.factor(test_set$label))
matrix_test_pca_five_step$byClass

########################################
######################################### upsampling binary MAX
set.seed(123)
training_index = createDataPartition(y = binary_class_max$class, 
                                     p = 0.75, list = FALSE)
training_set_binaryup = binary_class_max[training_index, ]
test_set_binaryup = binary_class_max[-training_index, ]

freqbinary <- training_set_binaryup %>%
  group_by(class) %>%
  summarise(counts = n())
ggplot(freqbinary, aes(x = class, y = counts)) +
  geom_bar(stat = "identity")

upbinary <- ovun.sample(class ~ ., data = training_set_binaryup,
                        method = "over", N = 140)$data
frequpbinary <- upbinary %>%
  group_by(class) %>%
  summarise(counts = n())

train_control <- trainControl (method = "repeatedcv",
                               number = 10,
                               repeats = 3,
                               summaryFunction = multiClassSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.752,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1,1.5))

svm_pca_upbinary <- train(class ~., data = upbinary,
                        method = "svmRadial",
                        preProcess = c("scale", "center"),
                        trControl = train_control,
                        tuneLength = 5,
                        tuneGrid = grid,
                        metric = "Accuracy")
plot(svm_pca_upbinary)
pred_train_pca_upbinary <- predict(svm_pca_upbinary, upbinary)
matrix_train_pca_upbinary <- confusionMatrix(pred_train_pca_upbinary, 
                                             as.factor(upbinary$class))
matrix_train_pca_upbinary$byClass
#predict
pred_test_pca_upbinary <- predict(svm_pca_upbinary, test_set_binaryup)
matrix_test_pca_upbinary <- confusionMatrix(pred_test_pca_upbinary, 
                                            as.factor(test_set_binaryup$class))
matrix_test_pca_upbinary$byClass

################################## upsampling 3-level MAX frames
set.seed(123)
training_index = createDataPartition(y = multiclass_max$class, 
                                     p = 0.75, list = FALSE)
training_set_multiup = multiclass_max[training_index, ]
test_set_multiup = multiclass_max[-training_index, ]

freqmulticlass <- training_set_multiup %>%
  group_by(class) %>%
  summarise(counts = n())
ggplot(freqmulticlass, aes(x = class, y = counts)) +
  geom_bar(stat = "identity")

upmultidata <- training_set_multiup %>%
  filter(class == "mild depressed" | class == "moderate to severe depressed")

upmulti <- ovun.sample(class ~ ., data = upmultidata,
                        method = "over", N = 88)$data

upmultidata2 <- training_set_multiup %>%
  filter(class == "moderate to severe depressed" | class == "non-depressed")

upmulti2 <- ovun.sample(class ~ ., data = upmultidata2,
                        method = "over", N = 88)$data

upmulti_complete <- full_join(upmulti, upmulti2)

frequpmulti <- upmulti_complete %>%
  group_by(class) %>%
  summarise(counts = n())

upmulti_complete$class <- factor(upmulti_complete$class, 
                                 c("non-depressed", "mild depressed", "moderate to severe depressed"))
test_set_multiup$class <- factor(test_set_multi$class, 
                                 c("non-depressed", "mild depressed", "moderate to severe depressed"))

upmulti_complete$label <- NULL

frequpmultitest <- test_set_multi %>%
  group_by(label) %>%
  summarise(counts = n())

train_control <- trainControl (method = "repeatedcv",
                               number = 10,
                               repeats = 3,
                               summaryFunction = multiClassSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75,2.0,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1.5))

svm_pca_upmulti <- train(class ~., data = upmulti_complete,
                          method = "svmRadial",
                          trControl = train_control,
                         preProcess = c("scale", "center"),
                         tuneGrid = grid,
                         tuneLength = 5,
                          metric = "Accuracy")
plot(svm_pca_upmulti)
pred_train_pca_multi <- predict(svm_pca_upmulti, upmulti_complete)
matrix_train_pca_multi <- confusionMatrix(pred_train_pca_multi, 
                                          as.factor(upmulti_complete$class))
matrix_train_pca_multi$byClass
#predict
pred_test_pca_upmulti <- predict(svm_pca_upmulti, test_set_multiup)
matrix_test_pca_upmulti <- confusionMatrix(pred_test_pca_upmulti, 
                                           as.factor(test_set_multiup$class))
matrix_test_pca_upmulti$byClass

############################################# binary and 3- level and regression max
set.seed(123)
training_index = createDataPartition(y = binary_class_max$class, 
                                     p = 0.75, list = FALSE)
training_set_binarymax = binary_class_max[training_index, ]
test_set_binarymax = binary_class_max[-training_index, ]

svm_binarymax <- train(class ~., data = training_set_binarymax,
                          method = "svmRadial",
                          preProcess = c("scale", "center"),
                          trControl = train_control,
                          tuneLength = 5,
                          tuneGrid = grid,
                          metric = "Accuracy")
pred_binarymax <- predict(svm_binarymax, training_set_binarymax)
matrix_binarymax <- confusionMatrix(pred_binarymax, 
                                    as.factor(training_set_binarymax$class))
matrix_binarymax$byClass
pred_test_binarymax <- predict(svm_binarymax, test_set_binarymax)
matrix_test_binarymax <- confusionMatrix(pred_test_binarymax, 
                                      as.factor(test_set_binarymax$class))
matrix_test_binarymax$byClass

set.seed(123)
training_index = createDataPartition(y = multiclass_max$class, 
                                     p = 0.75, list = FALSE)
training_set_multimax = multiclass_max[training_index, ]
test_set_multimax = multiclass_max[-training_index, ]

svm_multimax <- train(class ~., data = training_set_multimax,
                    method = "svmRadial",
                    preProcess = c("scale", "center"),
                    trControl = train_control,
                    tuneLength = 5,
                    tuneGrid = grid,
                    metric = "Accuracy")
pred_multimax <- predict(svm_multimax, training_set_multimax)
matrix_multimax <- confusionMatrix(pred_multimax, as.factor(training_set_multimax$class))
matrix_multimax$byClass
pred_test_multimax <- predict(svm_multimax, test_set_multimax)
matrix_test_multimax <- confusionMatrix(pred_test_multimax, 
                                        as.factor(test_set_multimax$class))
matrix_test_multimax$byClass

set.seed(123)
training_index = createDataPartition(y = regressionmax$hrsd.score, 
                                     p = 0.75, list = FALSE)
training_set_regressionmax = regressionmax[training_index, ]
test_set_regressionmax = regressionmax[-training_index, ]

train_control2 <- trainControl (method = "repeatedcv",
                                number = 10,
                                repeats = 3,
                                savePred = T)

svm_regressionmax <- train(hrsd.score ~., data = training_set_regressionmax,
                        method = "svmRadial",
                        preProcess = c("scale", "center"),
                        trControl = train_control2,
                        tuneLength = 5,
                        tuneGrid = grid)
svm_regressionmax$pred
pred_regressionmax <- predict(svm_regressionmax, training_set_regressionmax)
RMSEtrainmax <- postResample(pred_regressionmax, training_set_regressionmax$hrsd.score)
pred_test_regressionmax <- predict(svm_regressionmax, test_set_regressionmax)
RMSEtestmax <- postResample(pred_test_regressionmax, test_set_regressionmax$hrsd.score)

####################################################### regression max upsampling
set.seed(123)
training_index = createDataPartition(y = regressionmax$hrsd.score, 
                                     p = 0.75, list = FALSE)
training_set_regressionmax = regressionmax[training_index, ]
test_set_regressionmax = regressionmax[-training_index, ]

freq2 <- training_set_regressionmax %>%
  group_by(hrsd.score) %>%
  summarise(counts = n())
ggplot(freq2, aes(x = hrsd.score, y = counts)) +
  geom_bar(stat = "identity")

upsampling_regression <- SmoteRegress(hrsd.score ~., training_set_regressionmax,
                                      rel = "auto", thr.rel = 0.65,
                                      C.perc = list(1,3.0), k = 5)

freq3 <- upsampling_regression %>%
  group_by(hrsd.score) %>%
  summarise(counts = n())
ggplot(freq3, aes(x = hrsd.score, y = counts)) +
  geom_bar(stat = "identity")

svm_regressionmax <- train(hrsd.score ~., data = training_set_regressionmax,
                           method = "svmRadial",
                           preProcess = c("scale", "center"),
                           trControl = train_control2,
                           tuneLength = 5,
                           tuneGrid = grid)
svm_regressionmax$pred
pred_regression_max <- predict(svm_regressionmax, training_set_regressionmax)
RMSEtrainmax <- postResample(pred_regression_max, training_set_regressionmax$hrsd.score)
pred_test_regression_max <- predict(svm_regressionmax, test_set_regressionmax)
RMSEtestmax <- postResample(pred_test_regression_max, test_set_regressionmax$hrsd.score)

################################################# binary PCA MAX
label <- binary_class_max$class
binary_pca <- binary_class_max[-1]
pca <- princomp(na.omit(binary_pca), cor = TRUE, scores = TRUE)
fviz_eig(pca)
plot(pca)
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
pca_data <- predict(pca, binary_class_max)
pcadata <- as.data.frame(pca_data)
pcadata <- cbind(label, pcadata)
pcadata1 <- select(pcadata, 1:4)

set.seed(123)
training_index = createDataPartition(y = pcadata1$label, 
                                     p = 0.75, list = FALSE)
training_set_binary = pcadata1[training_index, ]
test_set_binary = pcadata1[-training_index, ]

train_control <- trainControl (method = "repeatedcv",
                               number = 10,
                               repeats = 3,
                               summaryFunction = multiClassSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.752,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1,1.5))

svm_pca_binary <- train(label ~., data = training_set_binary,
                          method = "svmRadial",
                          preProcess = c("scale", "center"),
                          trControl = train_control,
                          tuneLength = 5,
                          tuneGrid = grid,
                          metric = "Accuracy")
plot(svm_pca_binary)
pred_train_pca_binary <- predict(svm_pca_binary, training_set_binary)
matrix_train_pca_binary <- confusionMatrix(pred_train_pca_binary, as.factor(training_set_binary$label))
matrix_train_pca_binary$byClass
#predict
pred_test_pca_binary <- predict(svm_pca_binary, test_set_binary)
matrix_test_pca_binary <- confusionMatrix(pred_test_pca_binary, as.factor(test_set_binary$label))
matrix_test_pca_binary$byClass

####################################### multiclass PCA MAX
label <- multiclass_max$class
multi_pca <- multiclass_max[-1]
pca2 <- princomp(na.omit(multi_pca), cor = TRUE, scores = TRUE)
pca <- fviz_eig(pca2)
plot(pca2)
fviz_pca_var(pca2,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
pca_data2 <- predict(pca2, multiclass_max)
pcadata2 <- as.data.frame(pca_data2)
pcadata2 <- cbind(label, pcadata2)
pcadata2 <- select(pcadata2, 1:4)

set.seed(123)
training_index = createDataPartition(y = pcadata2$label, 
                                     p = 0.75, list = FALSE)
training_set_multi = pcadata2[training_index, ]
test_set_multi = pcadata2[-training_index, ]

train_control <- trainControl (method = "repeatedcv",
                               number = 10,
                               repeats = 3,
                               summaryFunction = multiClassSummary)
grid <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75,2,5),
                    sigma = c(0.0001,0.001,0.01,0.1,1,1.5))

svm_pca_multi <- train(label ~., data = training_set_multi,
                        method = "svmRadial",
                        preProcess = c("scale", "center"),
                        trControl = train_control,
                        tuneLength = 5,
                        tuneGrid = grid,
                        metric = "Accuracy")
plot(svm_pca_multi)
pred_train_pca_multi <- predict(svm_pca_multi, training_set_multi)
matrix_train_pca_multi <- confusionMatrix(pred_train_pca_multi, as.factor(training_set_multi$label))
matrix_train_pca_multi$byClass
#predict
pred_test_pca_multi <- predict(svm_pca_multi, test_set_multi)
matrix_test_pca_multi <- confusionMatrix(pred_test_pca_multi, as.factor(test_set_multi$label))
matrix_test_pca_multi$byClass

################################################ binary and 3-level and regression mean
set.seed(123)
training_index = createDataPartition(y = binary_class$class, 
                                     p = 0.75, list = FALSE)
training_set_binarymean = binary_class[training_index, ]
test_set_binarymean = binary_class[-training_index, ]

svm_binary <- train(class ~., data = training_set_binarymean,
                    method = "svmRadial",
                    preProcess = c("scale", "center"),
                    trControl = train_control,
                    tuneLength = 5,
                    tuneGrid = grid,
                    metric = "Accuracy")
pred_binary <- predict(svm_binary, training_set_binarymean)
matrix_binary <- confusionMatrix(pred_binary, as.factor(training_set_binarymean$class))
matrix_binary$byClass
pred_test_binary <- predict(svm_binary, test_set_binarymean)
matrix_test_binary <- confusionMatrix(pred_test_binary, 
                                      as.factor(test_set_binarymean$class))
matrix_test_binary$byClass

set.seed(123)
training_index = createDataPartition(y = multiclass$class, 
                                     p = 0.75, list = FALSE)
training_set_multi = multiclass[training_index, ]
test_set_multi = multiclass[-training_index, ]

svm_multi <- train(class ~., data = training_set_multi,
                   method = "svmRadial",
                   preProcess = c("scale", "center"),
                   trControl = train_control,
                   tuneLength = 5,
                   tuneGrid = grid,
                   metric = "Accuracy")
pred_multi <- predict(svm_multi, training_set_multi)
matrix_multi <- confusionMatrix(pred_multi, as.factor(training_set_multi$class))
matrix_multi$byClass
pred_test_multi <- predict(svm_multi, test_set_multi)
matrix_test_multi <- confusionMatrix(pred_test_multi, as.factor(test_set_multi$class))
matrix_test_multi$byClass

set.seed(123)
training_index = createDataPartition(y = regression$hrsd.score, 
                                     p = 0.75, list = FALSE)
training_set_regression = regression[training_index, ]
test_set_regression = regression[-training_index, ]

train_control2 <- trainControl (method = "repeatedcv",
                                number = 10,
                                repeats = 3,
                                savePred = T)

svm_regression <- train(hrsd.score ~., data = training_set_regression,
                           method = "svmRadial",
                           preProcess = c("scale", "center"),
                           trControl = train_control2,
                           tuneLength = 5,
                           tuneGrid = grid)
svm_regression$pred
pred_regression <- predict(svm_regression, training_set_regression)
RMSEtrain <- postResample(pred_regression, training_set_regression$hrsd.score)
pred_test_regression <- predict(svm_regression, test_set_regression)
RMSEtest <- postResample(pred_test_regression, test_set_regression$hrsd.score)

############## regression upsampling max
set.seed(123)
training_index = createDataPartition(y = regressionmaxup$hrsd.score, 
                                     p = 0.75, list = FALSE)
training_set_regression = regressionmaxup[training_index, ]
test_set_regression = regressionmaxup[-training_index, ]

freqmultiheadmove <- training_set_regression %>%
  group_by(group) %>%
  summarise(counts = n())

regression_maxup <- training_set_regression %>%
  filter(group == "mild depressed" | group == "moderate to severe depressed")

regression_maxup1 <- ovun.sample(group ~ ., data = regression_maxup,
                                method = "under", N = 50)$data

regression_maxup2 <- training_set_regression %>%
  filter(group == "moderate to severe depressed" | group == "non-depressed")

regression_maxup3 <- ovun.sample(group ~ ., data = regression_maxup2,
                                method = "under", N = 50)$data
regression_maxup4 <- regression_maxup3 %>%
  filter(group == "non-depressed")

regression_maxup_complete <- full_join(regression_maxup1, regression_maxup4)

freqmultiheadmove <- regression_maxup_complete %>%
  group_by(group) %>%
  summarise(counts = n())
mean(regression_maxup_complete$hrsd.score)

train_control2 <- trainControl (method = "repeatedcv",
                               number = 10,
                               repeats = 3)
regression_maxup_complete$group <- NULL
svm_regression <- train(hrsd.score ~., data = regression_maxup_complete,
                        method = "svmRadial",
                        preProcess = c("scale", "center"),
                        trControl = train_control2,
                        tuneLength = 5,
                        tuneGrid = grid)
svm_regression$bestTune
svm_regression$pred$pred
pred_regression <- predict(svm_regression, regression_maxup_complete)
RMSEtrain <- postResample(pred_regression, regression_maxup_complete$hrsd.score)
pred_test_regression <- predict(svm_regression, test_set_regression)
RMSEtest <- postResample(pred_test_regression, test_set_regression$hrsd.score)

################################################# 
################## binary 
label <- binary_class$class
binary_pca <- binary_class[-1]
pca <- princomp(na.omit(binary_pca), cor = TRUE, scores = TRUE)
fviz_eig(pca)
plot(pca)
fviz_pca_var(pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
pca_data <- predict(pca, binary_class)
pcadata <- as.data.frame(pca_data)
pcadata <- cbind(label, pcadata)
pcadata1 <- select(pcadata, 1:4)

set.seed(123)
training_index = createDataPartition(y = pcadata1$label, 
                                     p = 0.75, list = FALSE)
training_set_binary = pcadata1[training_index, ]
test_set_binary = pcadata1[-training_index, ]

freqbinary <- training_set_binary %>%
  group_by(label) %>%
  summarise(counts = n())
ggplot(freqbinary, aes(x = label, y = counts)) +
  geom_bar(stat = "identity")

upbinary <- ovun.sample(label ~ ., data = training_set_binary,
                        method = "over", N = 140)$data
frequpbinary <- upbinary %>%
  group_by(label) %>%
  summarise(counts = n())

train_control <- trainControl (method = "repeatedcv",
                               number = 10,
                               repeats = 5,
                               summaryFunction = multiClassSummary)
lrbinary <- train(label ~., data = upbinary,
                  trControl = train_control,
                  method = "glm")
lrbinary_pred_train <- predict(lrbinary, upbinary)
matrix_lrbinary_pred_train <- confusionMatrix(lrbinary_pred_train, 
                                              as.factor(upbinary$label))
lrbinary_pred_test <- predict(lrbinary, test_set_binary)
matrix_lrbinary_pred_test <- confusionMatrix(lrbinary_pred_test, 
                                             as.factor(test_set_binary$label))

####################### multi
label <- multiclass$class
multi_pca <- multiclass[-1]
pca2 <- princomp(na.omit(multi_pca), cor = TRUE, scores = TRUE)
fviz_eig(pca2)
plot(pca2)
fviz_pca_var(pca2,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
pca_data2 <- predict(pca2, multiclass)
pcadata2 <- as.data.frame(pca_data2)
pcadata2 <- cbind(label, pcadata2)
pcadata2 <- select(pcadata2, 1:4)

set.seed(123)
training_index = createDataPartition(y = pcadata2$label, 
                                     p = 0.75, list = FALSE)
training_set_multi = pcadata2[training_index, ]
test_set_multi = pcadata2[-training_index, ]

freqmulticlass <- training_set_multi %>%
  group_by(label) %>%
  summarise(counts = n())
ggplot(freqmulticlass, aes(x = label, y = counts)) +
  geom_bar(stat = "identity")

upmultidata <- training_set_multi %>%
  filter(label == "mild depressed" | label == "moderate to severe depressed")

upmulti <- ovun.sample(label ~ ., data = upmultidata,
                       method = "over", N = 88)$data

upmultidata2 <- training_set_multi %>%
  filter(label == "moderate to severe depressed" | label == "non-depressed")

upmulti2 <- ovun.sample(label ~ ., data = upmultidata2,
                        method = "over", N = 88)$data

upmulti_complete <- full_join(upmulti, upmulti2)

frequpmulti <- upmulti_complete %>%
  group_by(label) %>%
  summarise(counts = n())

upmulti_complete$label <- factor(upmulti_complete$label, 
                                 c("non-depressed", "mild depressed", "moderate to severe depressed"))
test_set_multi$label <- factor(test_set_multi$label, 
                               c("non-depressed", "mild depressed", "moderate to severe depressed"))

frequpmultitest <- test_set_multi %>%
  group_by(label) %>%
  summarise(counts = n())

train_control <- trainControl (method = "repeatedcv",
                               number = 10,
                               repeats = 5)

lrmulti <- train(label ~., data = upmulti_complete,
                  trControl = train_control,
                  method = "rf")

lrmulti_pred_train <- predict(lrmulti, upmulti_complete)
matrix_lrmulti_pred_train <- confusionMatrix(lrmulti_pred_train, 
                                              as.factor(upmulti_complete$label))
lrmulti_pred_test <- predict(lrmulti, test_set_multi)
matrix_lrmulti_pred_test <- confusionMatrix(lrmulti_pred_test, 
                                             as.factor(test_set_multi$label))
matrix_lrmulti_pred_test$byClass

############################################################ LATE FUSION BINARY with upsampling
## FACEMOVE
set.seed(123)
training_index = createDataPartition(y = binary_facemove$class, 
                                     p = 0.60, list = FALSE)
training_set_binaryfacemove = binary_facemove[training_index, ]
test_set_binaryfacemove = binary_facemove[-training_index, ]

training_index2 = createDataPartition(y = test_set_binaryfacemove$class, 
                                     p = 0.50, list = FALSE)
validation_set_binaryfacemove = test_set_binaryfacemove[training_index2, ]
test_set_binaryfacemove = test_set_binaryfacemove[-training_index2, ]

freqbinary <- training_set_binaryfacemove %>%
  group_by(class) %>%
  summarise(counts = n())
ggplot(freqbinary, aes(x = class, y = counts)) +
  geom_bar(stat = "identity")

upbinaryfacemove <- ovun.sample(class ~ ., data = training_set_binaryfacemove,
                        method = "over", N = 112)$data
frequpbinaryfacemove <- upbinaryfacemove %>%
  group_by(class) %>%
  summarise(counts = n())

svm_binaryfacemove <- train(class ~., data = upbinaryfacemove,
                       method = "svmRadial",
                       preProcess = c("scale", "center"),
                       trControl = train_control,
                       tuneLength = 5,
                       tuneGrid = grid,
                       metric = "Accuracy")
pred_binaryfacemove <- predict(svm_binaryfacemove, upbinaryfacemove)
matrix_binaryfacemove <- confusionMatrix(pred_binaryfacemove, 
                                    as.factor(upbinaryfacemove$class))
matrix_binaryfacemove$byClass
pred_val_binaryfacemove <- predict(svm_binaryfacemove, validation_set_binaryfacemove)
matrix_val_binaryfacemove <- confusionMatrix(pred_val_binaryfacemove, 
                                         as.factor(validation_set_binaryfacemove$class))
matrix_val_binaryfacemove$byClass

## HEADMOVE
set.seed(123)
training_index = createDataPartition(y = binary_headmove$class, 
                                     p = 0.60, list = FALSE)
training_set_binaryheadmove = binary_headmove[training_index, ]
test_set_binaryheadmove = binary_headmove[-training_index, ]

training_index = createDataPartition(y = test_set_binaryheadmove$class, 
                                     p = 0.50, list = FALSE)
validation_set_binaryheadmove = test_set_binaryheadmove[training_index, ]
test_set_binaryheadmove = test_set_binaryheadmove[-training_index, ]

freqbinaryheadmove <- training_set_binaryheadmove %>%
  group_by(class) %>%
  summarise(counts = n())
ggplot(freqbinary, aes(x = class, y = counts)) +
  geom_bar(stat = "identity")

upbinaryheadmove <- ovun.sample(class ~ ., data = training_set_binaryheadmove,
                                method = "over", N = 112)$data

svm_binaryheadmove <- train(class ~., data = upbinaryheadmove,
                            method = "svmRadial",
                            preProcess = c("scale", "center"),
                            trControl = train_control,
                            tuneLength = 5,
                            tuneGrid = grid,
                            metric = "Accuracy")
pred_binaryheadmove <- predict(svm_binaryheadmove, upbinaryheadmove)
matrix_binaryheadmove <- confusionMatrix(pred_binaryheadmove, 
                                         as.factor(upbinaryheadmove$class))
matrix_binaryheadmove$byClass
pred_val_binaryheadmove <- predict(svm_binaryheadmove, validation_set_binaryheadmove)
matrix_val_binaryheadmove <- confusionMatrix(pred_val_binaryheadmove, 
                                              as.factor(validation_set_binaryheadmove$class))
matrix_val_binaryheadmove$byClass

## HEADPOSE
set.seed(123)
training_index = createDataPartition(y = binary_headpose$class, 
                                     p = 0.60, list = FALSE)
training_set_binaryheadpose = binary_headpose[training_index, ]
test_set_binaryheadpose = binary_headpose[-training_index, ]

training_index = createDataPartition(y = test_set_binaryheadpose$class, 
                                     p = 0.50, list = FALSE)
validation_set_binaryheadpose = test_set_binaryheadpose[training_index, ]
test_set_binaryheadpose = test_set_binaryheadpose[-training_index, ]

freqbinaryheadpose <- training_set_binaryheadpose %>%
  group_by(class) %>%
  summarise(counts = n())
ggplot(freqbinary, aes(x = class, y = counts)) +
  geom_bar(stat = "identity")

upbinaryheadpose <- ovun.sample(class ~ ., data = training_set_binaryheadpose,
                                method = "over", N = 112)$data

svm_binaryheadpose <- train(class ~., data = upbinaryheadpose,
                            method = "svmRadial",
                            preProcess = c("scale", "center"),
                            trControl = train_control,
                            tuneLength = 5,
                            tuneGrid = grid,
                            metric = "Accuracy")
pred_binaryheadpose <- predict(svm_binaryheadpose, upbinaryheadpose)
matrix_binaryheadpose <- confusionMatrix(pred_binaryheadpose, 
                                         as.factor(upbinaryheadpose$class))
matrix_binaryheadpose$byClass
pred_val_binaryheadpose <- predict(svm_binaryheadpose, validation_set_binaryheadpose)
matrix_val_binaryheadpose <- confusionMatrix(pred_val_binaryheadpose, 
                                              as.factor(validation_set_binaryheadpose$class))
matrix_val_binaryheadpose$byClass

## VOCAL
set.seed(123)
training_index = createDataPartition(y = binary_vocal$class, 
                                     p = 0.60, list = FALSE)
training_set_binaryvocal = binary_vocal[training_index, ]
test_set_binaryvocal = binary_vocal[-training_index, ]

training_index = createDataPartition(y = test_set_binaryvocal$class, 
                                     p = 0.50, list = FALSE)
validation_set_binaryvocal = test_set_binaryvocal[training_index, ]
test_set_binaryvocal = test_set_binaryvocal[-training_index, ]

freqbinaryvocal <- training_set_binaryvocal %>%
  group_by(class) %>%
  summarise(counts = n())
ggplot(freqbinary, aes(x = class, y = counts)) +
  geom_bar(stat = "identity")

upbinaryvocal <- ovun.sample(class ~ ., data = training_set_binaryvocal,
                                method = "over", N = 112)$data
frequpbinaryvocal <- upbinaryvocal %>%
  group_by(class) %>%
  summarise(counts = n())

svm_binaryvocal <- train(class ~., data = upbinaryvocal,
                            method = "svmRadial",
                            preProcess = c("scale", "center"),
                            trControl = train_control,
                            tuneLength = 5,
                            tuneGrid = grid,
                            metric = "Accuracy")
pred_binaryvocal <- predict(svm_binaryvocal, upbinaryvocal)
matrix_binaryvocal <- confusionMatrix(pred_binaryvocal, 
                                         as.factor(upbinaryvocal$class))
matrix_binaryvocal$byClass
pred_val_binaryvocal <- predict(svm_binaryvocal, validation_set_binaryvocal)
matrix_val_binaryvocal <- confusionMatrix(pred_val_binaryvocal, 
                                              as.factor(validation_set_binaryvocal$class))
matrix_val_binaryvocal$byClass

## fusing
test_set_binaryvocal$pred_vocal <- predict(svm_binaryvocal, test_set_binaryvocal)
test_set_binaryvocal$pred_headpose <- predict(svm_binaryheadpose, 
                                              test_set_binaryheadpose)
test_set_binaryvocal$pred_headmove <- predict(svm_binaryheadmove, 
                                              test_set_binaryheadmove)
test_set_binaryvocal$pred_facemove <- predict(svm_binaryfacemove,
                                              test_set_binaryfacemove)

test_set_binaryvocal <- test_set_binaryvocal %>%
  mutate(pred_vocal1 = case_when(.$pred_vocal == "depressed" ~ 1,
                                 .$pred_vocal == "non-depressed" ~ 0))
test_set_binaryvocal <- test_set_binaryvocal %>%
  mutate(pred_headpose1 = case_when(.$pred_headpose == "depressed" ~ 1,
                                 .$pred_headpose == "non-depressed" ~ 0))
test_set_binaryvocal <- test_set_binaryvocal %>%
  mutate(pred_headmove1 = case_when(.$pred_headmove == "depressed" ~ 1,
                                 .$pred_headmove == "non-depressed" ~ 0))
test_set_binaryvocal <- test_set_binaryvocal %>%
  mutate(pred_facemove1 = case_when(.$pred_facemove == "depressed" ~ 1,
                                 .$pred_facemove == "non-depressed" ~ 0))

test_set_binaryvocal$average <- (test_set_binaryvocal$pred_vocal1 * 0.30) + 
  (test_set_binaryvocal$pred_facemove1 * 0.20) + 
  (test_set_binaryvocal$pred_headmove1 * 0.35) + 
  (test_set_binaryvocal$pred_headpose1 * 0.15)

test_set_binaryvocal <- test_set_binaryvocal %>%
  mutate(pred = case_when(.$average >= 0.5 ~ "depressed",
                          .$average < 0.5 ~ "non-depressed"))

matrix_binaryfusion <- confusionMatrix(as.factor(test_set_binaryvocal$pred), 
                                       as.factor(test_set_binaryvocal$class))
matrix_binaryfusion$byClass

##################################################### LATE FUSION 3-LEVEL
## FACEMOVE
set.seed(123)
training_index = createDataPartition(y = multi_facemove$class, 
                                     p = 0.60, list = FALSE)
training_set_multifacemove = multi_facemove[training_index, ]
test_set_multifacemove = multi_facemove[-training_index, ]

training_index = createDataPartition(y = test_set_multifacemove$class, 
                                     p = 0.50, list = FALSE)
validation_set_multifacemove = test_set_multifacemove[training_index, ]
test_set_multifacemove = test_set_multifacemove[-training_index, ]

freqmulti <- training_set_multifacemove %>%
  group_by(class) %>%
  summarise(counts = n())
ggplot(freqbinary, aes(x = class, y = counts)) +
  geom_bar(stat = "identity")

upmultifacemove <- training_set_multifacemove %>%
  filter(class == "mild depressed" | class == "moderate to severe depressed")

upmultifacemove1 <- ovun.sample(class ~ ., data = upmultifacemove,
                       method = "over", N = 70)$data

upmultifacemove2 <- training_set_multifacemove %>%
  filter(class == "moderate to severe depressed" | class == "non-depressed")

upmultifacemove3 <- ovun.sample(class ~ ., data = upmultifacemove2,
                        method = "over", N = 70)$data

upmultifacemove_complete <- full_join(upmultifacemove1, upmultifacemove3)

svm_multifacemove <- train(class ~., data = upmultifacemove_complete,
                            method = "svmRadial",
                            preProcess = c("scale", "center"),
                            trControl = train_control,
                            tuneLength = 5,
                            tuneGrid = grid,
                            metric = "Accuracy")
pred_multifacemove <- predict(svm_multifacemove, upmultifacemove_complete)
matrix_multifacemove <- confusionMatrix(pred_multifacemove, 
                                         as.factor(upmultifacemove_complete$class))
matrix_multifacemove$byClass
pred_val_multifacemove <- predict(svm_multifacemove, validation_set_multifacemove)
matrix_val_multifacemove <- confusionMatrix(pred_val_multifacemove, 
                                              as.factor(validation_set_multifacemove$class))
matrix_val_multifacemove$byClass

## HEADMOVE
set.seed(123)
training_index = createDataPartition(y = multi_headmove$class, 
                                     p = 0.60, list = FALSE)
training_set_multiheadmove = multi_headmove[training_index, ]
test_set_multiheadmove = multi_headmove[-training_index, ]

training_index = createDataPartition(y = test_set_multiheadmove$class, 
                                     p = 0.50, list = FALSE)
validation_set_multiheadmove = test_set_multiheadmove[training_index, ]
test_set_multiheadmove = test_set_multiheadmove[-training_index, ]

freqmultiheadmove <- training_set_multiheadmove %>%
  group_by(class) %>%
  summarise(counts = n())

upmultiheadmove <- training_set_multiheadmove %>%
  filter(class == "mild depressed" | class == "moderate to severe depressed")

upmultiheadmove1 <- ovun.sample(class ~ ., data = upmultiheadmove,
                                method = "over", N = 70)$data

upmultiheadmove2 <- training_set_multiheadmove %>%
  filter(class == "moderate to severe depressed" | class == "non-depressed")

upmultiheadmove3 <- ovun.sample(class ~ ., data = upmultiheadmove2,
                                method = "over", N = 70)$data

upmultiheadmove_complete <- full_join(upmultiheadmove1, upmultiheadmove3)

svm_multiheadmove <- train(class ~., data = upmultiheadmove_complete,
                            method = "svmRadial",
                            preProcess = c("scale", "center"),
                            trControl = train_control,
                            tuneLength = 5,
                            tuneGrid = grid,
                            metric = "Accuracy")
pred_multiheadmove <- predict(svm_multiheadmove, upmultiheadmove_complete)
matrix_multiheadmove <- confusionMatrix(pred_multiheadmove, 
                                         as.factor(upmultiheadmove_complete$class))
matrix_multiheadmove$byClass
pred_val_multiheadmove <- predict(svm_multiheadmove, validation_set_multiheadmove)
matrix_val_multiheadmove <- confusionMatrix(pred_val_multiheadmove, 
                                              as.factor(validation_set_multiheadmove$class))
matrix_val_multiheadmove$byClass

## HEADPOSE
set.seed(123)
training_index = createDataPartition(y = multi_headpose$class, 
                                     p = 0.60, list = FALSE)
training_set_multiheadpose = multi_headpose[training_index, ]
test_set_multiheadpose = multi_headpose[-training_index, ]

training_index = createDataPartition(y = test_set_multiheadpose$class, 
                                     p = 0.50, list = FALSE)
validation_set_multiheadpose = test_set_multiheadpose[training_index, ]
test_set_multiheadpose = test_set_multiheadpose[-training_index, ]

freqmultiheadpose <- training_set_multiheadpose %>%
  group_by(class) %>%
  summarise(counts = n())

upmultiheadpose <- training_set_multiheadpose %>%
  filter(class == "mild depressed" | class == "moderate to severe depressed")

upmultiheadpose1 <- ovun.sample(class ~ ., data = upmultiheadpose,
                                method = "over", N = 70)$data

upmultiheadpose2 <- training_set_multiheadpose %>%
  filter(class == "moderate to severe depressed" | class == "non-depressed")

upmultiheadpose3 <- ovun.sample(class ~ ., data = upmultiheadpose2,
                                method = "over", N = 70)$data

upmultiheadpose_complete <- full_join(upmultiheadpose1, upmultiheadpose3)

svm_multiheadpose <- train(class ~., data = upmultiheadpose_complete,
                            method = "svmRadial",
                            preProcess = c("scale", "center"),
                            trControl = train_control,
                            tuneLength = 5,
                            tuneGrid = grid,
                            metric = "Accuracy")
pred_multiheadpose <- predict(svm_multiheadpose, upmultiheadpose_complete)
matrix_multiheadpose <- confusionMatrix(pred_multiheadpose, 
                                         as.factor(upmultiheadpose_complete$class))
matrix_multiheadpose$byClass
pred_val_multiheadpose <- predict(svm_multiheadpose, validation_set_multiheadpose)
matrix_val_multiheadpose <- confusionMatrix(pred_val_multiheadpose, 
                                              as.factor(validation_set_multiheadpose$class))
matrix_test_multiheadpose$byClass

## VOCAL
set.seed(123)
training_index = createDataPartition(y = multi_vocal$class, 
                                     p = 0.60, list = FALSE)
training_set_multivocal = multi_vocal[training_index, ]
test_set_multivocal = multi_vocal[-training_index, ]

training_index = createDataPartition(y = test_set_multivocal$class,
                                     p = 0.50, list = FALSE)
validation_set_multivocal = test_set_multivocal[training_index, ]
test_set_multivocal = test_set_multivocal[-training_index, ]

freqmultivocal <- training_set_multivocal %>%
  group_by(class) %>%
  summarise(counts = n())

upmultivocal <- training_set_multivocal %>%
  filter(class == "mild depressed" | class == "moderate to severe depressed")

upmultivocal1 <- ovun.sample(class ~ ., data = upmultivocal,
                                method = "over", N = 70)$data

upmultivocal2 <- training_set_multivocal %>%
  filter(class == "moderate to severe depressed" | class == "non-depressed")

upmultivocal3 <- ovun.sample(class ~ ., data = upmultivocal2,
                                method = "over", N = 70)$data

upmultivocal_complete <- full_join(upmultivocal1, upmultivocal3)

svm_multivocal <- train(class ~., data = upmultivocal_complete,
                         method = "svmRadial",
                         preProcess = c("scale", "center"),
                         trControl = train_control,
                         tuneLength = 5,
                         tuneGrid = grid,
                         metric = "Accuracy")
pred_multivocal <- predict(svm_multivocal, upmultivocal_complete)
matrix_multivocal <- confusionMatrix(pred_multivocal, 
                                      as.factor(upmultivocal_complete$class))
matrix_multivocal$byClass
pred_val_multivocal <- predict(svm_multivocal, validation_set_multivocal)
matrix_val_multivocal <- confusionMatrix(pred_val_multivocal, 
                                           as.factor(validation_set_multivocal$class))
matrix_test_multivocal$byClass

## fusing
test_set_multivocal$pred_vocal <- predict(svm_multivocal, test_set_multivocal)
test_set_multivocal$pred_headpose <- predict(svm_multiheadpose, 
                                              test_set_multiheadpose)
test_set_multivocal$pred_headmove <- predict(svm_multiheadmove, 
                                              test_set_multiheadmove)
test_set_multivocal$pred_facemove <- predict(svm_multifacemove,
                                              test_set_multifacemove)

test_set_multivocal <- test_set_multivocal %>%
  mutate(pred_vocal1 = case_when(.$pred_vocal == "moderate to severe depressed" ~ 2,
                                 .$pred_vocal == "mild depressed" ~ 1,
                                 .$pred_vocal == "non-depressed" ~ 0))
test_set_multivocal <- test_set_multivocal %>%
  mutate(pred_headpose1 = case_when(.$pred_headpose == "moderate to severe depressed" ~ 2,
                                    .$pred_headpose == "mild depressed" ~ 1,
                                    .$pred_headpose == "non-depressed" ~ 0))
test_set_multivocal <- test_set_multivocal %>%
  mutate(pred_headmove1 = case_when(.$pred_headmove == "moderate to severe depressed" ~ 2,
                                    .$pred_headmove == "mild depressed" ~ 1,
                                    .$pred_headmove == "non-depressed" ~ 0))
test_set_multivocal <- test_set_multivocal %>%
  mutate(pred_facemove1 = case_when(.$pred_facemove == "moderate to severe depressed" ~ 2,
                                    .$pred_facemove == "mild depressed" ~ 1,
                                    .$pred_facemove == "non-depressed" ~ 0))

test_set_multivocal$average <- (test_set_multivocal$pred_vocal1 * 0.15) + 
  (test_set_multivocal$pred_facemove1 * 0.40) + 
  (test_set_multivocal$pred_headmove1 * 0.30) + 
  (test_set_multivocal$pred_headpose1 * 0.15)

test_set_multivocal <- test_set_multivocal %>%
  mutate(pred = case_when(.$average >= 1.5 ~ "moderate to severe depressed",
                          .$average >= 1.0 & .$average < 1.5 ~ "mild depressed",
                          .$average < 1.0 ~ "non-depressed"))

matrix_multifusion <- confusionMatrix(as.factor(test_set_multivocal$pred), 
                                       as.factor(test_set_multivocal$class))
matrix_multifusion$byClass


