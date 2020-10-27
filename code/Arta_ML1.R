# see https://www.tmwr.org/ for a guide
# more general overview https://lgatto.github.io/IntroMachineLearningWithR/index.html

# Packages

# resampling, splitting and validation
library(rsample)
# feature, engineering, preprocessing
library(recipes)
#specifying models
library(parsnip)
#tuning
library(tune)
#tuning parameters
library(dials)
#performance measurements
library(yardstick)
#variable importance plots
library(vip)
# combining feature engineering and model specification
library(workflows)
# data manipulation
library(dplyr)
# plotting
library(ggplot2)
#parallelism
library(doFuture)
library(parallel)
# timing
library(tictoc)

# OR
# library(tidymodels)

# Data #####

data(credit_data, package = "modeldata")
credit_data <- credit_data %>% as_tibble()
credit_data

# we are going to model if your credit is good based on data

# EDA ####
ggplot(credit_data, aes(x = Status)) + geom_bar()

ggplot (credit_data, aes(x = Status, y = Amount)) + geom_violin()

ggplot (credit_data, aes(x = Status, y = Age)) + geom_violin()

ggplot (credit_data, aes(x = Status, y = Income)) + geom_violin()

ggplot (credit_data, aes(x = Status, y = Income)) + geom_violin(draw_quantiles = 0.5)

ggplot(credit_data, aes(x = Age, y = Income, color = Status)) + geom_point()

ggplot(credit_data, aes(x = Age, y = Income, color = Status)) + geom_point() + facet_wrap(~
        Status) + theme(legend.position = 'none')

ggplot(credit_data, aes(x = Age, y = Income, color = Status)) + geom_hex() + facet_wrap(~
        Status) + theme(legend.position = 'none')

# Split Data ####
set.seed(42069)

# will split data into testing and training
credit_split <-
    initial_split(credit_data, prop = 0.8, strata = 'Status')

credit_split

#  Train/Test/Total
# <Analysis/Assess/Total>
# <3565/889/4454>

credit_split %>% class()

train <- training(credit_split)
test <- testing(credit_split)

train
train %>% glimpse()
train %>% class()

# Skimr displays most of the numerical attributes from summary, but it also displays missing values, more quantile information and an inline histogram for each variable
library(skimr)
skim(train)



# Feature Engineering ####
# Food themes <- wow!!!
# from recipes
# Max Kuhn wrote caret, which was the way to do unified ML but then he wrote tidymodels

# Inputs: predictors; x; features; covariates; independent variable; data; attributes; descriptors
# Outcome; response; y; label; target; dependent variable; output; result; known; true

table(credit_data$Status)

# parents will get put into misc column below
# step_novel does the same thing - step_other is preferable
cat_train_1 <- c("rent", "own", "mortgage")
cat_test_1 <- c("rent", "parents")

cat_train_2 <- c("rent", "own", "mortgage")
cat_test_2 <- c("rent", "own")

# dot means "everything else", so
# below means status by all other columns
rec1 <- recipe(Status ~ ., data = train) %>%
    # xgboost can handle
    step_downsample(Status, under_ratio = 1.2) %>%
    # not really needed for xgboost
    step_normalize(Age, Price) %>%
    # step_other lumps together outlier variables
    # step_other(Home, Marital, Records, Jobs, other='Misc')
    # this below is a shortcut to above
    step_other(all_nominal(), -Status, other = 'Misc') %>%
    # remove columns with very little variability, near-zero variance
    step_nzv(all_predictors()) %>%
    # mode imputation - fills in blanks for columns
    # xgboost doesn't need imputation so can remove later
    step_modeimpute(all_nominal(), -Status) %>%
    # imputes missing using nearest neighbors - sometimes has trouble with nominal vars so sometimes use mode on nominal and knn on numeric
    step_knnimpute(all_numeric()) %>%
    step_dummy(all_nominal(), -Status, one_hot = TRUE)

rec1

# downsampling "evens out" quantity between dataset
# can also use step_smote (which might be better)
# step_normalize(Age) would subtract the mean and div by std dev for a column

# Model Specification ####

# Boosted trees are decision trees but more than one and they learn from the performance/outcomes of previous trees

# From parsnip by Max Kuhn
xg_spec1 <- boost_tree(mode = 'classification') %>%
    set_engine('xgboost')

xg_spec1

boost_tree(mode = 'classification') %>% set_engine('C5.0')
boost_tree(mode = 'classification') %>% set_engine('spark')

# BART: dbart
# catboost
# LightGBM

xg_spec1 <-
    boost_tree(mode = "classification", trees = 100) %>% set_engine("xgboost")
xg_spec1

# parsnip gives us a uniform naming convention for all of the parameters! further examples

linear_reg() %>% set_engine('lm')
linear_reg(penalty = 0.826) %>% set_engine('glmnet')
linear_reg() %>% set_engine('keras')
linear_reg() %>% set_engine('stan')
linear_reg() %>% set_engine('spark')

rand_forest() %>% set_engine('randomForest')
rand_forest() %>% set_engine('ranger')

# Build Workflow ####

# prep calculates stats and formulate actions and bake executes steps
prepped <- rec1 %>% prep() %>% bake(new_data = NULL)
prepped

fit0 <- fit(xg_spec1, Status ~ ., data = prepped)
fit0

# above is obsolete, workflows automatically preps and bakes data:

# combine feature engineering and model specification into one step

flow1 <- workflow() %>%
    add_recipe(rec1) %>%
    add_model(xg_spec1)

flow1

# Fit Our Model
# we are feeding the training data through our workflow
fit1 <- fit(flow1, data = train)
fit1
fit1 %>% class()

# variable importance plot
fit1 %>% extract_model() %>% vip()
fit1 %>% extract_model() %>% xgboost::xgb.plot.multi.trees()

# to save without any of the computations
# readr::write_rds(fit1, 'fit1.rds')
# to save with computations to run on a edge computer
# xgboost::xgb.save(fit1 %>% extract_model(), fname = "xg1.model")

# How did we do? ####

# accuracy, logloss, AUC

# from yardstick pkg, metric_set will compute different accuracy metrics

loss_fn <- metric_set(accuracy, mn_log_loss, roc_auc)
loss_fn
#        metric        class direction
# 1    accuracy class_metric  maximize
# 2 mn_log_loss  prob_metric  minimize
# 3     roc_auc  prob_metric  maximize

# we need a validation dataset

## two types of validation
# train and validation sets
# or cross validation

# we are going to split our training data again
# one part "sub-training" and other validation

# from rsample
val_split <- validation_split(data = train,
    prop = 0.8,
    strata = 'Status')

val_split

# # Validation Set Split (0.8/0.2)  using stratification
# # A tibble: 1 x 2
#   splits             id
#   <list>             <chr>
# 1 <split [2.9K/712]> validation

val_split$splits[[1]]
# <Training/Validation/Total>
# <2853/712/3565>

credit_split %>% class()
# [1] "rsplit"   "mc_split"

val_split$splits[[1]] %>% class()
# [1] "val_split" "rsplit"

val_split %>% class()
# [1] "validation_split" "rset"
# [3] "tbl_df"           "tbl"
# [5] "data.frame"

# with one fxn, you can take val_split, train and validate on component sets and look at different metrics

# resamples can be validation split, bootstramp (resample) or cross validation

# fitting a model, testing it and error checking
val1 <-
    fit_resamples(object = flow1,
        resamples = val_split,
        metrics = loss_fn)
val1

val1 %>% collect_metrics()
# # A tibble: 3 x 5
#   .metric     .estimator  mean     n std_err
#   <chr>       <chr>      <dbl> <int>   <dbl>
# 1 accuracy    binary     0.742     1      NA
# 2 mn_log_loss binary     0.651     1      NA
# 3 roc_auc     binary     0.806     1      NA

val1$.metrics
# # A tibble: 3 x 3
#   .metric     .estimator .estimate
#   <chr>       <chr>          <dbl>
# 1 accuracy    binary         0.742
# 2 mn_log_loss binary         0.651
# 3 roc_auc     binary         0.806

# a cross validation includes a "test fold" or a "bin" for each iteration in your training set to produce error scores and then calculates the mean error

install.packages("animation")
library(animation)
cv.ani(k = 10)

cv_split <- vfold_cv(data = train, v = 10, strata = "Status")
cv_split
cv_split %>% class() # very similar to val_split
val_split %>% class()

cv_split$splits[[1]]

# we can also do reapeated cv, e.g. 10 times
vfold_cv(
    data = train,
    v = 10,
    strata = "Status",
    repeats = 3
)

cv_split <- vfold_cv(
    data = train,
    v = 5,
    strat = "Status",
    repeats = 2
)
cv_split

# let's say our 1x10 had a bad shuffling, 2x5 gives us more robustness

# same fxn as from the regular validation (val1)
cv1 <-
    fit_resamples(object = flow1,
        resamples = cv_split,
        metrics = loss_fn)

cv1

# ten measures of each of these metrics
cv1$.metrics[[1]]
cv1$.metrics[[2]]
cv1$.metrics[[3]]

cv1 %>% collect_metrics()
# # A tibble: 3 x 5
#   .metric     .estimator  mean     n std_err
#   <chr>       <chr>      <dbl> <int>   <dbl>
# 1 accuracy    binary     0.739    10 0.00832
# 2 mn_log_loss binary     0.637    10 0.0176
# 3 roc_auc     binary     0.798    10 0.00776

# for a multi-class set, you cannot use roc_auc but that's it

# More Parameters ####

xg_spec2 <-
    boost_tree(mode = "classification", trees = 300) %>% set_engine("xgboost")

# Let's not do this
# workflow() %>%
#     add_model(xg_spec2)%>%
#     add_recipe(rec1)

# you can update your model
flow2 <- flow1 %>%
    update_model(xg_spec2)

flow2

val2 <-
    fit_resamples(flow2, resamples = val_split, metrics = loss_fn)
val2
val2 %>% collect_metrics()

xg_spec3 <-
    boost_tree("classification", trees = 300, learn_rate = 0.2) %>% set_engine("xgboost")
xg_spec3

flow3 <- flow2 %>% update_model(xg_spec3)

val3 <-
    fit_resamples(flow3, resamples = val_split, metrics = loss_fn)
val3 %>% collect_metrics()

xg_spec4 <-
    boost_tree(
        "classification",
        trees = 300,
        learn_rate = 0.2,
        sample_size = 0.5
    ) %>%
    set_engine("xgboost")
xg_spec4

flow4 <- flow3 %>% update_model(xg_spec4)
val4 <-
    fit_resamples(flow3, resamples = val_split, metrics = loss_fn)
val4 %>% collect_metrics()

# Missing Data ####

rec2 <- recipe(Status ~ ., data = train) %>%
    step_nzv(all_predictors()) %>%
    step_other(all_nominal(), -Status, other = "Misc") %>%
    themis::step_downsample(Status, under_ratio = 1.2) %>%
    step_dummy(all_nominal(), -Status, one_hot = TRUE)
rec2

flow5 <- flow4 %>%
    update_recipe(rec2)
flow5

val5 <-
    fit_resamples(flow5, resamples = val_split, metrics = loss_fn)
val5 %>% collect_metrics()
val4 %>% collect_metrics()

val5$.notes

# Imbalanced Data ####

rec3 <- recipe(Status ~ ., data = train) %>%
    step_nzv(all_predictors()) %>%
    step_other(all_nominal(), -Status, other = "Misc") %>%
    step_dummy(all_nominal(), -Status, one_hot = TRUE)
rec3

flow6 <- flow5 %>% update_recipe(rec3)

val6 <-
    fit_resamples(flow6, resamples = val_split, metrics = loss_fn)

val5 %>% collect_metrics()
val6 %>% collect_metrics()

table(train$Status)

# if you give xgboost the ratio of bad to good rows, it'll help with the fitting
scaler <- train %>% count(Status) %>% pull(n) %>% purrr::reduce(`/`)

xg_spec5 <-
    boost_tree(
        'classification',
        trees = 300,
        learn_rate = 0.2,
        sample_size = 0.5
    ) %>% set_engine("xgboost", scale_pos_weight = !!(1 / scaler))

# tidymodels reverses the ratio for some reason, so we have to do the inverse of scaler

# not all engines take the same tuning parameter
# scale_pos_weight is xgboost specific and is good for tuning
# scalar is a variable, R uses lazy evaluation, so !! means it gets computed on the spot

flow7 <- flow6 %>%
    update_model(xg_spec5)
flow7

val7 <-
    fit_resamples(flow7, resamples = val_split, metrics = loss_fn)
val7 %>% collect_metrics()
val6 %>% collect_metrics()
val5 %>% collect_metrics()

# Tune Parameters ####

# from tune

xg_spec6 <-
    boost_tree(
        'classification',
        learn_rate = 0.2,
        sample_size = 0.5,
        trees = tune()
    ) %>%
    set_engine("xgboost", scale_pos_weight = !!(1 / scaler))
xg_spec6

flow8 <- flow7 %>%
    update_model(xg_spec6)
flow8

# does not work
# fit7 <- fit(flow7, data=train)

# does not work
# val8 <- fit_resamples(flow8, resamples=val_split, metrics=loss_fn)
# val8$.notes

# pros and cons of validate vs cross-validate
# cv - pros: more robust
#      cons: expensive

# Set up parallelism
# from doFuture and parallel
registerDoFuture()

# I have a 12 core cpu on my desktop otherwise I wouldn't recommend
cl <- makeCluster(6)
plan(cluster, workers = cl)

options(tidymodels.dark = TRUE)
#from tictoc
tic()
#tune
tune8_val <- tune_grid(
    flow8,
    resamples = val_split,
    grid = 20,
    metrics = loss_fn,
    control = control_grid(verbose = TRUE, allow_par = TRUE)
)
toc()

tune8_val
tune8_val$.notes
tune8_val$.metrics

tune8_val %>% collect_metrics()

tune8_val %>% show_best(metric = "roc_auc")

tic()
tune8_cv <- tune_grid(
    flow8,
    resamples = cv_split,
    grid = 20,
    metrics = loss_fn,
    control = control_grid(verbose = TRUE, allow_par = TRUE)
)
toc()

tune8_cv
tune8_cv$.metrics[[1]]$trees %>% unique

tune8_cv %>% collect_metrics()
tune8_cv %>% autoplot()
tune8_cv %>% show_best(metric = 'roc_auc')

# Other Tuning Parameters ####

xg_spec7 <- boost_tree(
    "classification",
    trees = tune(),
    learn_rate = 0.2,
    sample_size = tune(),
    tree_depth = tune()
) %>%
    set_engine("xgboost", scale_pos_weight = !!(1 / scaler))
xg_spec7

# tree_depth controls how many subsequent "follow up questions" get asked

flow9 <- flow8 %>%
    update_model(xg_spec7)
flow9

flow9 %>% parameters()
flow9 %>% parameters() %>% class()
flow9 %>% parameters() %>% pull(object)

# from dials
trees()
trees(range = c(10, 300))

tree_depth()
tree_depth(range = c(2, 8))

sample_size()
sample_prop()
sample_prop(c(0.3, 0.8))

params9 <- flow9 %>%
    parameters() %>%
    update(
        trees = trees(range = c(10, 300)),
        tree_depth = tree_depth(range = c(2, 8)),
        sample_size = sample_prop(range = c(0.3, 0.8))
    )
params9
params9 %>% pull(object)

tic()
val9 <- tune_grid(
    flow9,
    resamples = val_split,
    grid = 40,
    metrics = loss_fn,
    control = control_grid(verbose = TRUE, allow_par = TRUE),
    param_info = params9
)
toc()
val9

val9

val9 %>% show_best(metric = 'roc_auc')
val9 %>% autoplot(metric = 'roc_auc')

grid10 <- grid_max_entropy(params9, size = 40)
grid10

tic()
val10 <- tune_grid(
    flow9,
    resamples = val_split,
    grid = grid10,
    metrics = loss_fn,
    control = control_grid(verbose = TRUE, allow_par = TRUE)
)
toc()

val10 %>% collect_metrics()
val10 %>% show_best(metric = 'roc_auc', n = 10)

val10 %>% select_best(metric = 'roc_auc')

boost_tree(
    "classification",
    trees = 127,
    tree_depth = 2,
    sample_size = 0.509
)

# Finalize Model ####

mod10 <-
    flow9 %>% finalize_workflow(val10 %>% select_best(metric = 'roc_auc'))
flow9
mod10

val10.1 <-
    fit_resamples(mod10, resamples = val_split, metrics = loss_fn)
val10.1 %>% collect_metrics()
val10 %>% show_best()
val10 %>% collect_metrics()
val10.1 %>% collect_metrics()

test

# Last Fit ####

results10 <- last_fit(mod10, split = credit_split, metrics = loss_fn)
results10 %>% collect_metrics()

# Make Predictions ####

# Fit the model on All the data
# Predict on some new data (pretend "test" is new)

fit10 <- fit(mod10, data=credit_data)
fit10 %>% extract_model() %>% vip()

# Predict on some new data (pretend "test" is new)
preds10 <- predict(fit10, new_data = test)
preds10

# fit is for fitting one model with set parameters
# fit_resamples is for fitting multiple models for validation w set parameters
# tune_grid is for tuning over tuning parameters

# What we want is the probability that each thing will be one or the other thing
preds10_prob <- predict(fit10, new_data = test, type = "prob")
preds10_prob




