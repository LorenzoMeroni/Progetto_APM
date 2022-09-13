####### LIBRERIE #########
library(tictoc)
library(tidymodels)
library(tidyverse)
tidymodels_prefer()
library(modeldata)
library(ggpubr)
library(ggplot2)
library(data.table)       
library(stringr)
library(ggcorrplot)
library(e1071)
library(xgboost)
library(stacks)
library(baguette)
library(MASS)
#### 0. Import data ####
setwd("C:/Users/lorenzo/OneDrive/Desktop/Data Mining")
train <- read.csv2("Home_prices_train.csv", header=T,sep=";")
test <- read.csv2("Home_prices_test.csv", header=T,sep=";")
#### 1. Data split  ####
Sale_train <- train %>%
  transmute(
    date_sold = str_replace_all(date_sold, "T00:00:00Z", ""),
    price,
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    sqft_above,
    sqft_basement,
    nn_sqft_living,
    nn_sqft_lot,    
    floors,
    waterfront,
    view,
    agebuilt = as.integer(format(Sys.Date(), "%Y")) - train$yr_built,
    agerenovated = as.integer(format(Sys.Date(), "%Y")) - train$year_renovated,
    zip_code,
    lattitude,
    longitude,
    nn_sqft = nn_sqft_living + nn_sqft_lot
    ) 
last_sell = Sale_train$date_sold %>% max(na.rm=TRUE)
Sale_train$day_since_last_sell = as.numeric(as.Date(last_sell)-as.Date(Sale_train$date_sold))
Sale_train <- Sale_train[,-1] %>% mutate_if(is.character,factor)



Sale_test <- test %>%
  transmute(
    date_sold = str_replace_all(date_sold, "T00:00:00Z", ""),
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    sqft_above,
    sqft_basement,
    nn_sqft_living,
    nn_sqft_lot,      
    floors,
    waterfront,
    view,
    agebuilt = as.integer(format(Sys.Date(), "%Y")) - test$yr_built,
    agerenovated = as.integer(format(Sys.Date(), "%Y")) - test$year_renovated,
    zip_code,
    lattitude,
    longitude,
    nn_sqft = nn_sqft_living + nn_sqft_lot 
  ) 
last_sell = Sale_test$date_sold %>% max(na.rm=TRUE)
Sale_test$day_since_last_sell = as.numeric(as.Date(last_sell)-as.Date(Sale_test$date_sold))
Sale_test <- Sale_test[,-1] %>% mutate_if(is.character,factor)

#### 2. Cross-validation ####
set.seed(1502)
house_cv_folds <- 
  vfold_cv(Sale_train, strata = price)
#### 3. Model ####
linear_reg_spec <- 
  linear_reg() %>% 
  set_engine("lm")

rf_spec <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

xgb_spec <- 
  boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(),mtry = tune(), 
             min_n = tune(), sample_size = tune(), trees = 700) %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")
#### 4. workflow #######
#no pre processing
model_vars <- 
  workflow_variables(outcomes = price, 
                     predictors = everything())
no_pre_proc <- 
  workflow_set(
    preproc = list(simple = model_vars), 
    models = list(Rf = rf_spec, 
                  xgboost = xgb_spec)
  )
#splines
linear_rec <- 
  recipe(price ~ ., data = Sale_train) %>% 
  step_ns(lattitude, deg_free = 15) %>% 
  step_ns(longitude, deg_free = 20)

with_splines <- 
  workflow_set(
    preproc = list(linear = linear_rec), 
    models = list(linear_reg = linear_reg_spec)
  )

all_workflows <- 
  bind_rows(with_splines,no_pre_proc) %>% 
  # Make the workflow ID's a little more simple: 
  mutate(wflow_id = gsub("(linear_)|(simple_)", "", wflow_id))
all_workflows

#### 5. Model tuning #######
grid_ctrl <-
  control_grid(
    save_pred = TRUE,
    parallel_over = "everything",
    save_workflow = TRUE
  )
library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)
grid_results <-
  all_workflows %>%
  workflow_map(
    seed = 123,
    resamples = house_cv_folds,
    grid = 50,
    control = grid_ctrl,
    verbose = TRUE,
    fn="tune_grid",
    metrics = metric_set(rmse,mae),
  )

grid_results %>% 
  rank_results() %>% 
  filter(.metric == "mae") %>% 
  select(model, .config, mae = mean, rank)

#### 6. Prediction ####

Sale_stack <- stacks() %>% 
  add_candidates(grid_results) 

set.seed(2002)
ens <- blend_predictions(Sale_stack, 
                         penalty = 10^seq(-2, -0.5, length = 20))
ens
autoplot(ens, "weights") +
  geom_text(aes(x = weight + 0.01, label = model), hjust = 0) + 
  theme(legend.position = "none") +
  lims(x = c(-0.01, 0.8))
ens <- fit_members(ens)
reg_metrics <- metric_set(mae)
ens_train_pred <- 
  predict(ens, Sale_train[,-1]) 
ens_train_pred %>% 
  reg_metrics(Sale_train[,1], .pred)
#Sul training se l'errore è sottostimato

ens_test_pred <- 
  predict(ens, Sale_test) 
write.table(file="875319_previsione.txt", ens_test_pred$.pred, row.names = FALSE, col.names = FALSE)
