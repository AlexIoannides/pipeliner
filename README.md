
<!-- README.md is generated from README.Rmd. Please edit that file -->
<!-- [![codecov](https://codecov.io/github/alexioannides/pipeliner/branch/master/graphs/badge.svg)](https://codecov.io/github/alexioannides/pipeliner) -->
[![Build Status](https://travis-ci.org/AlexIoannides/pipeliner.svg?branch=master)](https://travis-ci.org/AlexIoannides/pipeliner) [![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/AlexIoannides/pipeliner?branch=master&svg=true)](https://ci.appveyor.com/project/AlexIoannides/pipeliner) [![cran version](http://www.r-pkg.org/badges/version/pipeliner)](https://cran.r-project.org/package=pipeliner) <!--[![rstudio mirror downloads](http://cranlogs.r-pkg.org/badges/grand-total/pipeliner)](https://github.com/metacran/cranlogs.app)-->

Machine Learning Pipelines for R
================================

Building machine learning and statistical models often requires pre- and post-transformation of the input and/or response variables, prior to training (or fitting) the models. For example, a model may require training on the logarithm of the response and input variables. As a consequence, fitting and then generating predictions from these models requires repeated application of transformation and inverse-transformation functions - to go from the domain of the original input variables to the domain of the original output variables (via the model). This is usually quite a laborious and repetitive process that leads to messy code and notebooks.

The `pipeliner` package aims to provide an elegant solution to these issues by implementing a common interface and workflow with which it is possible to:

-   define transformation and inverse-transformation functions;
-   fit a model on training data; and then,
-   generate a prediction (or model-scoring) function that automatically applies the entire pipeline of transformations and inverse-transformations to the inputs and outputs of the inner-model and its predicted values (or scores).

The idea of pipelines is inspired by the machine learning pipelines implemented in [Apache Spark's MLib library](http://spark.apache.org/docs/latest/ml-pipeline.html "Pipelines in Apache Spark MLib") (which are in-turn inspired by Python's scikit-Learn package). This package is still in its infancy and the latest development version can be downloaded from [this GitHub repository](https://github.com/AlexIoannides/pipeliner "Pipeliner on GitHub") using the `devtools` package (bundled with RStudio),

``` r
devtools::install_github("alexioannides/pipeliner")
```

Pipes in the Pipleline
----------------------

There are currently four types of pipeline section - a section being a function that wraps a user-defined function - that can be assembled into a pipeline:

-   `transform_features`: wraps a function that maps input variables (or features) to another space - e.g.,

``` r
transform_features(function(df) { 
  data.frame(x1 = log(df$var1))
})
```

-   `transform_response`: wraps a function that maps the response variable to another space - e.g.,

``` r
transform_response(function(df) { 
  data.frame(y = log(df$response))
})
```

-   `estimate_model`: wraps a function that defines how to estimate a model from training data in a data.frame - e.g.,

``` r
estimate_model(function(df) { 
  lm(y ~ 1 + x1, df)
})
```

-   `inv_transform_features(f)`: wraps a function that is the inverse to `transform_response`, such that we can map from the space of inner-model predictions to the one of output domain predictions - e.g.,

``` r
inv_transform_response(function(df) { 
  data.frame(pred_response = exp(df$pred_y))
})
```

As demonstrated above, each one of these functions expects as its argument another unary function of a data.frame (i.e. it has to be a function of a single data.frame). With the **exception** of `estimate_model`, which expects the input function to return an object that has a `predict.object-class-name` method existing in the current environment (e.g. `predict.lm` for linear models built using `lm()`), all the other transform functions also expect their input functions to return data.frames (consisting entirely of columns **not** present in the input data.frame). If any of these rules are violated then appropriately named errors will be thrown to help you locate the issue.

If this sounds complex and convoluted then I encourage you to to skip to the examples below - this framework is **very** simple to use in practice. Simplicity is the key aim here.

Two Interfaces to Rule Them All
-------------------------------

I am a great believer and protagonist for functional programming - especially for data-related tasks like building machine learning models. At the same time the notion of a 'machine learning pipeline' is well represented with a simple object-oriented class hierarchy (which is how it is implemented in [Apache Spark's](http://spark.apache.org/docs/latest/ml-pipeline.html "Pipelines in Apache Spark MLib")). I couldn't decide which style of interface was best, so I implemented both within `pipeliner` (using the same underlying code) and ensured their output can be used interchangeably. To keep this introduction simple, however, I'm only going to talk about the functional interface - those interested in the (more) object-oriented approach are encouraged to read the manual pages for the `ml_pipeline_builder` 'class'.

### Example Usage with a Functional Flavour

We use the `faithful` dataset shipped with R, together with the `pipeliner` package to estimate a linear regression model for the eruption duration of 'Old Faithful' as a function of the inter-eruption waiting time. The transformations we apply to the input and response variables - before we estimate the model - are simple scaling by the mean and standard deviation (i.e. mapping the variables to z-scores).

The end-to-end process for building the pipeline, estimating the model and generating in-sample predictions (that include all interim variable transformations), is as follows,

``` r
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
```

    ##   eruptions waiting         x1           y pred_model pred_eruptions
    ## 1     3.600      79  0.5960248  0.09831763  0.5369058       4.100592
    ## 2     1.800      54 -1.2428901 -1.47873278 -1.1196093       2.209893
    ## 3     3.333      74  0.2282418 -0.13561152  0.2056028       3.722452
    ## 4     2.283      62 -0.6544374 -1.05555759 -0.5895245       2.814917
    ## 5     4.533      85  1.0373644  0.91575542  0.9344694       4.554360
    ## 6     2.883      55 -1.1693335 -0.52987412 -1.0533487       2.285521

### Accessing Inner Models & Prediction Functions

We can access the estimated inner models directly and compute summaries, etc - for example,

``` r
summary(lm_pipeline$inner_model)
```

    ## 
    ## Call:
    ## lm(formula = y ~ 1 + x1, data = df)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -1.13826 -0.33021  0.03074  0.30586  1.04549 
    ## 
    ## Coefficients:
    ##               Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -3.139e-16  2.638e-02    0.00        1    
    ## x1           9.008e-01  2.643e-02   34.09   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.435 on 270 degrees of freedom
    ## Multiple R-squared:  0.8115, Adjusted R-squared:  0.8108 
    ## F-statistic:  1162 on 1 and 270 DF,  p-value: < 2.2e-16

Pipeline prediction functions can also be accessed directly in a similar way - for example,

``` r
pred_function <- lm_pipeline$predict 
predictions <- pred_function(data, verbose = FALSE)

head(predictions)
```

    ##   pred_eruptions
    ## 1       4.100592
    ## 2       2.209893
    ## 3       3.722452
    ## 4       2.814917
    ## 5       4.554360
    ## 6       2.285521

Turbo-Charged Pipelines in the Tidyverse
----------------------------------------

The `pipeliner` approach to building models becomes even more concise when combined with the set of packages in the [tidyverse](http://tidyverse.org "Welcome to The Tidyverse!"). For example, the 'Old Faithful' pipeline could be rewritten as,

``` r
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
```

    ## [1] 4.100592 2.209893 3.722452 2.814917 4.554360 2.285521

Nice, compact and expressive (if I don't say so myself)!

### Compact Cross-validation

If we now introduce the `modelr` package into this workflow and adopt the the list-columns pattern described in Hadley Wickham's [R for Data Science](http://r4ds.had.co.nz/many-models.html#list-columns-1 "R 4 Data Science - Many Models & List Columns"), we can also achieve wonderfully compact end-to-end model estimation and cross-validation,

``` r
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
```

    ## # A tibble: 1 × 2
    ##   mean_rmse    sd_rmse
    ##       <dbl>      <dbl>
    ## 1 0.4874391 0.01783045

Forthcoming Attractions
=======================

I built `pipeliner` largely to fill a hole in my own workflows. Up until now I've used Max Kuhn's excellent [caret package](http://topepo.github.io/caret/index.html "Caret") quite a bit, but for in-the-moment model building (e.g. within a R Notebook) it wasn't simplifying the code *that* much, and the style doesn't quite fit with the tidy and functional world that I now inhabit most of the time. So, I plugged the hole by myself. I intend to live with `pipeliner` for a while to get an idea of where it might go next, but I am always open to suggestions (and bug notifications) - please [leave any ideas here](https://github.com/AlexIoannides/pipeliner/issues "Pipeliner Issues on GitHub").
