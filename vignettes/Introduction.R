## ---- eval=FALSE, include=TRUE-------------------------------------------
#  devtools::install_github("alexioannides/pipeliner")

## ---- eval=FALSE, include=TRUE-------------------------------------------
#  transform_features(function(df) {
#    data.frame(x1 = log(df$var1))
#  })

## ---- eval=FALSE, include=TRUE-------------------------------------------
#  transform_response(function(df) {
#    data.frame(y = log(df$response))
#  })

## ---- eval=FALSE, include=TRUE-------------------------------------------
#  estimate_model(function(df) {
#    lm(y ~ 1 + x1, df)
#  })

## ---- eval=FALSE, include=TRUE-------------------------------------------
#  inv_transform_response(function(df) {
#    data.frame(pred_response = exp(df$pred_y))
#  })

## ------------------------------------------------------------------------
library(pipeliner)

data <- faithful

lm_pipeline <- pipeline(
  data,
  
  transform_features(function(df) { 
    data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))
  }),
  
  transform_response(function(df) {
    data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
  }),
  
  estimate_model(function(df) { 
    lm(y ~ 1 + x1, df)
  }),
  
  inv_transform_response(function(df) { 
    data.frame(pred_eruptions = df$pred_model * sd(df$eruptions) + mean(df$eruptions))
  })
)

in_sample_predictions <- predict(lm_pipeline, data, verbose = TRUE)  
head(in_sample_predictions)

## ------------------------------------------------------------------------
summary(lm_pipeline$inner_model)

## ------------------------------------------------------------------------
pred_function <- lm_pipeline$predict 
predictions <- pred_function(data, verbose = FALSE)

head(predictions)

## ---- warning=FALSE, message=FALSE---------------------------------------
library(tidyverse)

lm_pipeline <- data %>% 
  pipeline(
    transform_features(function(df) { 
      transmute(df, x1 = (waiting - mean(waiting)) / sd(waiting))
    }),
    
    transform_response(function(df) {
      transmute(df, y = (eruptions - mean(eruptions)) / sd(eruptions))
    }),
    
    estimate_model(function(df) { 
      lm(y ~ 1 + x1, df)
    }),
    
    inv_transform_response(function(df) { 
      transmute(df, pred_eruptions = pred_model * sd(eruptions) + mean(eruptions))
    })
  )

head(predict(lm_pipeline, data))

## ------------------------------------------------------------------------
library(modelr)

# define a function that estimates a machine learning pipeline on a single fold of the data
pipeline_func <- function(df) {
  pipeline(
    df,
    transform_features(function(df) {
      transmute(df, x1 = (waiting - mean(waiting)) / sd(waiting))
    }),

    transform_response(function(df) {
      transmute(df, y = (eruptions - mean(eruptions)) / sd(eruptions))
    }),

    estimate_model(function(df) {
      lm(y ~ 1 + x1, df)
    }),

    inv_transform_response(function(df) {
      transmute(df, pred_eruptions = pred_model * sd(eruptions) + mean(eruptions))
    })
  )
}

# 5-fold cross-validation using machine learning pipelines
cv_rmse <- crossv_kfold(data, 5) %>% 
  mutate(model = map(train, ~ pipeline_func(as.data.frame(.x))),
         predictions = map2(model, test, ~ predict(.x, as.data.frame(.y))),
         residuals = map2(predictions, test, ~ .x - as.data.frame(.y)$eruptions),
         rmse = map_dbl(residuals, ~ sqrt(mean(.x ^ 2)))) %>% 
  summarise(mean_rmse = mean(rmse), sd_rmse = sd(rmse))

cv_rmse

