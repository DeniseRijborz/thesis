## late fusion binary without upsampling
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

svm_binaryfacemove <- train(class ~., data = training_set_binaryfacemove,
                            method = "svmRadial",
                            preProcess = c("scale", "center"),
                            trControl = train_control,
                            tuneLength = 5,
                            tuneGrid = grid,
                            metric = "Accuracy")
pred_binaryfacemove <- predict(svm_binaryfacemove, training_set_binaryfacemove)
matrix_binaryfacemove <- confusionMatrix(pred_binaryfacemove, 
                                         as.factor(training_set_binaryfacemove$class))
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

svm_binaryheadmove <- train(class ~., data = training_set_binaryheadmove,
                            method = "svmRadial",
                            preProcess = c("scale", "center"),
                            trControl = train_control,
                            tuneLength = 5,
                            tuneGrid = grid,
                            metric = "Accuracy")
pred_binaryheadmove <- predict(svm_binaryheadmove, training_set_binaryheadmove)
matrix_binaryheadmove <- confusionMatrix(pred_binaryheadmove, 
                                         as.factor(training_set_binaryheadmove$class))
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

svm_binaryheadpose <- train(class ~., data = training_set_binaryheadpose,
                            method = "svmRadial",
                            preProcess = c("scale", "center"),
                            trControl = train_control,
                            tuneLength = 5,
                            tuneGrid = grid,
                            metric = "Accuracy")
pred_binaryheadpose <- predict(svm_binaryheadpose, training_set_binaryheadpose)
matrix_binaryheadpose <- confusionMatrix(pred_binaryheadpose, 
                                         as.factor(training_set_binaryheadpose$class))
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

svm_binaryvocal <- train(class ~., data = training_set_binaryvocal,
                         method = "svmRadial",
                         preProcess = c("scale", "center"),
                         trControl = train_control,
                         tuneLength = 5,
                         tuneGrid = grid,
                         metric = "Accuracy")
pred_binaryvocal <- predict(svm_binaryvocal, training_set_binaryvocal)
matrix_binaryvocal <- confusionMatrix(pred_binaryvocal, 
                                      as.factor(training_set_binaryvocal$class))
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

test_set_binaryvocal$average <- (test_set_binaryvocal$pred_vocal1 * 0.25) + 
  (test_set_binaryvocal$pred_facemove1 * 0.30) + 
  (test_set_binaryvocal$pred_headmove1 * 0.25) + 
  (test_set_binaryvocal$pred_headpose1 * 0.20)

test_set_binaryvocal <- test_set_binaryvocal %>%
  mutate(pred = case_when(.$average >= 0.5 ~ "depressed",
                          .$average < 0.5 ~ "non-depressed"))

matrix_binaryfusion <- confusionMatrix(as.factor(test_set_binaryvocal$pred), 
                                       as.factor(test_set_binaryvocal$class))
test_set_binaryvocal$pred <- factor(test_set_binaryvocal$pred, 
                                 c("depressed", "non-depressed"))
test_set_binaryvocal$class <- factor(test_set_binaryvocal$class, 
                               c("depressed", "non-depressed"))

matrix_binaryfusion$byClass

######### late fusion 3-level without upsampling
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

svm_multifacemove <- train(class ~., data = training_set_multifacemove,
                           method = "svmRadial",
                           preProcess = c("scale", "center"),
                           trControl = train_control,
                           tuneLength = 5,
                           tuneGrid = grid,
                           metric = "Accuracy")
pred_multifacemove <- predict(svm_multifacemove, training_set_multifacemove)
matrix_multifacemove <- confusionMatrix(pred_multifacemove, 
                                        as.factor(training_set_multifacemove$class))
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


svm_multiheadmove <- train(class ~., data = training_set_multiheadmove,
                           method = "svmRadial",
                           preProcess = c("scale", "center"),
                           trControl = train_control,
                           tuneLength = 5,
                           tuneGrid = grid,
                           metric = "Accuracy")
pred_multiheadmove <- predict(svm_multiheadmove, training_set_multiheadmove)
matrix_multiheadmove <- confusionMatrix(pred_multiheadmove, 
                                        as.factor(training_set_multiheadmove$class))
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

svm_multiheadpose <- train(class ~., data = training_set_multiheadpose,
                           method = "svmRadial",
                           preProcess = c("scale", "center"),
                           trControl = train_control,
                           tuneLength = 5,
                           tuneGrid = grid,
                           metric = "Accuracy")
pred_multiheadpose <- predict(svm_multiheadpose, training_set_multiheadpose)
matrix_multiheadpose <- confusionMatrix(pred_multiheadpose, 
                                        as.factor(training_set_multiheadpose$class))
matrix_multiheadpose$byClass
pred_val_multiheadpose <- predict(svm_multiheadpose, validation_set_multiheadpose)
matrix_val_multiheadpose <- confusionMatrix(pred_val_multiheadpose, 
                                             as.factor(validation_set_multiheadpose$class))
matrix_val_multiheadpose$byClass

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

svm_multivocal <- train(class ~., data = training_set_multivocal,
                        method = "svmRadial",
                        preProcess = c("scale", "center"),
                        trControl = train_control,
                        tuneLength = 5,
                        tuneGrid = grid,
                        metric = "Accuracy")
pred_multivocal <- predict(svm_multivocal, training_set_multivocal)
matrix_multivocal <- confusionMatrix(pred_multivocal, 
                                     as.factor(training_set_multivocal$class))
matrix_multivocal$byClass
pred_val_multivocal <- predict(svm_multivocal, validation_set_multivocal)
matrix_val_multivocal <- confusionMatrix(pred_val_multivocal, 
                                          as.factor(validation_set_multivocal$class))
matrix_val_multivocal$byClass

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

test_set_multivocal$average <- (test_set_multivocal$pred_vocal1 * 0.10) + 
  (test_set_multivocal$pred_facemove1 * 0.60) + 
  (test_set_multivocal$pred_headmove1 * 0.20) + 
  (test_set_multivocal$pred_headpose1 * 0.10)

test_set_multivocal <- test_set_multivocal %>%
  mutate(pred = case_when(.$average >= 1.5 ~ "moderate to severe depressed",
                          .$average >= 1.0 & .$average < 1.5 ~ "mild depressed",
                          .$average < 1.0 ~ "non-depressed"))

matrix_multifusion <- confusionMatrix(as.factor(test_set_multivocal$pred), 
                                      as.factor(test_set_multivocal$class))
matrix_multifusion$byClass

