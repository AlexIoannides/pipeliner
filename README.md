
<!-- README.md is generated from README.Rmd. Please edit that file -->
<!-- [![codecov](https://codecov.io/github/alexioannides/pipeliner/branch/master/graphs/badge.svg)](https://codecov.io/github/alexioannides/pipeliner) -->
[![Build Status](https://travis-ci.org/AlexIoannides/pipeliner.svg?branch=master)](https://travis-ci.org/AlexIoannides/pipeliner) [![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/AlexIoannides/pipeliner?branch=master&svg=true)](https://ci.appveyor.com/project/AlexIoannides/pipeliner) <!--[![cran version](http://www.r-pkg.org/badges/version/pipeliner)](https://cran.r-project.org/package=pipeliner) [![rstudio mirror downloads](http://cranlogs.r-pkg.org/badges/grand-total/pipeliner)](https://github.com/metacran/cranlogs.app)-->

Machine Learning Pipelines for R
================================

Building machine learning models often requires pre- and post-transformation of the input and/or response variables, prior to training (or fitting) the models. For example, a model may require training on the logarithm of the response and input variables. As a consequence, fitting and then generating predictions from these models requires repeated application of transformation and inverse-transformation functions, to go from the original input to original output variables (via the model).

This package is inspired by the machine learning pipelines used in Apache Spark, and provides a common interface with which it is possible to:

-   define transformation and inverse-transformation functions;
-   fit a model on training data; and then,
-   generate a prediction (or model-scoring) function that automatically applies the entire pipeline of transformation and inverse-transformation to the inputs and outputs of the inner-model's predicted scores.

Example Usage - OO API
----------------------

We use the `faithful` dataset shipped with R, together with the `pipeliner` package to estimate a linear regression model for the eruption duration of Old Faithful as a function of the inter-eruption waiting time (duration). The transformations we apply to the input and response variables - before we estimate the model - are simple scaling by the mean and standard deviation (i.e. mapping the variables to z-scores).

``` r
library(pipeliner)

data <- faithful

lm_pipeline <- ml_pipline_builder()

lm_pipeline$transform_features(function(df) { 
  data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))
})

lm_pipeline$transform_response(function(df) {
  data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
})

lm_pipeline$inv_transform_response(function(df) { 
  data.frame(pred_eruptions = df$pred_model * sd(df$eruptions) + mean(df$eruptions))
})

lm_pipeline$estimate_model(function(df) { 
  lm(y ~ 0 + x1, df)
})

lm_pipeline$fit(data)
```

We can access the estimated inner model directly and compute summaries as usual.

``` r
summary(lm_pipeline$inner_model())
```

    ## 
    ## Call:
    ## lm(formula = y ~ 0 + x1, data = df)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -1.13826 -0.33021  0.03074  0.30586  1.04549 
    ## 
    ## Coefficients:
    ##    Estimate Std. Error t value Pr(>|t|)    
    ## x1  0.90081    0.02638   34.15   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.4342 on 271 degrees of freedom
    ## Multiple R-squared:  0.8115, Adjusted R-squared:  0.8108 
    ## F-statistic:  1166 on 1 and 271 DF,  p-value: < 2.2e-16

Now that the pipeline has been estimated it is easy to generate predictions without having to handle transformations explicitly.

``` r
in_sample_predictions <- lm_pipeline$predict(data)
head(in_sample_predictions)
```

    ##   eruptions waiting         x1           y pred_model pred_eruptions
    ## 1     3.600      79  0.5960248  0.09831763  0.5369058       4.100592
    ## 2     1.800      54 -1.2428901 -1.47873278 -1.1196093       2.209893
    ## 3     3.333      74  0.2282418 -0.13561152  0.2056028       3.722452
    ## 4     2.283      62 -0.6544374 -1.05555759 -0.5895245       2.814917
    ## 5     4.533      85  1.0373644  0.91575542  0.9344694       4.554360
    ## 6     2.883      55 -1.1693335 -0.52987412 -1.0533487       2.285521

Alternatively, we provde a `predict` method for estimated pipelines, that works like the `predict` methods for any other R model.

``` r
in_sample_predictions <- predict(lm_pipeline, data)
head(in_sample_predictions)
```

    ## [1] 3.600 1.800 3.333 2.283 4.533 2.883

Example Usage - Functional API
------------------------------

As a functional alternative to the above,

``` r
library(magrittr)

lm_pipeline <- data %>% 
  pipeline(
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

more_predictions <- predict(lm_pipeline, verbose = TRUE, data)  
head(more_predictions)
```

    ##   eruptions waiting         x1           y pred_model pred_eruptions
    ## 1     3.600      79  0.5960248  0.09831763  0.5369058       4.100592
    ## 2     1.800      54 -1.2428901 -1.47873278 -1.1196093       2.209893
    ## 3     3.333      74  0.2282418 -0.13561152  0.2056028       3.722452
    ## 4     2.283      62 -0.6544374 -1.05555759 -0.5895245       2.814917
    ## 5     4.533      85  1.0373644  0.91575542  0.9344694       4.554360
    ## 6     2.883      55 -1.1693335 -0.52987412 -1.0533487       2.285521
