# Copyright 2016 Alex Ioannides
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


context('api')


# ---- transform_features ----
test_that("transform_features yields a function that transforms features as expected", {
  # arrange
  data <- faithful
  f <- transform_features(function(df) {
    data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))
  })

  # act
  f_out <- f(data)

  # assert
  output <- cbind(data, x1 = (data$waiting - mean(data$waiting)) / sd(data$waiting))
  expect_true(is.function(f))
  expect_s3_class(f, "transform_features")
  expect_s3_class(f, "ml_pipeline_section")
  expect_equal(f_out, output)
})


test_that("transform_features throws errors with invalid inputs", {
  # arrange
  data <- faithful
  f1 <- function(df1, df2) {
    data.frame(x1 = (df1$waiting - mean(df1$waiting)) / sd(df1$waiting))
  }
  f2 <- function(df) {
    data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))[[1]]
  }
  f3 <- data.frame(x1 = (data$waiting - mean(data$waiting)) / sd(data$waiting))[[1]]

  # act & assert
  expect_error(transform_features(f1))
  expect_error(transform_features(f2)(data))
  expect_error(transform_features(f3))
})


# ---- transform_response ----
test_that("transform_response yields a function that transforms response vars as expected", {
  # arrange
  data <- faithful
  f <- transform_response(function(df) {
    data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
  })

  # act
  f_out <- f(data)

  # assert
  output <- cbind(data, y = (data$eruptions - mean(data$eruptions)) / sd(data$eruptions))
  expect_true(is.function(f))
  expect_s3_class(f, "transform_response")
  expect_s3_class(f, "ml_pipeline_section")
  expect_equal(f_out, output)
})


test_that("transform_response throws errors with invalid inputs", {
  # arrange
  data <- faithful
  f1 <- function(df1, df2) {
    data.frame(y = (df1$eruptions - mean(df1$eruptions)) / sd(df1$eruptions))
  }
  f2 <- function(df) {
    data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))[[1]]
  }
  f3 <- data.frame(y = (data$eruptions - mean(data$eruptions)) / sd(data$eruptions))[[1]]

  # act & assert
  expect_error(transform_features(f1))
  expect_error(transform_features(f2)(data))
  expect_error(transform_features(f3))
})


# ---- estimate_model ----
test_that("estimate_model yields a function that estimates a model as expected", {
  # arrange
  data <- faithful
  f <- estimate_model(function(df) {
    lm(eruptions ~ 1 + waiting, df)
  })

  # act
  f_out <- f(data)

  # assert
  df <- data
  output <- lm(eruptions ~ 1 + waiting, df)

  expect_true(is.function(f))
  expect_s3_class(f, "estimate_model")
  expect_s3_class(f, "ml_pipeline_section")
  expect_s3_class(f_out, "lm")
  expect_equal(f_out, output)
})


test_that("inv_transform_response throws errors with invalid inputs", {
  # arrange
  data <- faithful
  f1 <- function(df1) {
    model <- lm(eruptions ~ 1 + waiting, df)
    class(model) <- NULL
    model
  }
  f2 <- function(df1, df2) {
    lm(eruptions ~ 1 + waiting, df)
  }
  f3 <- function(df) {
    data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))[[1]]
  }
  f4 <- data.frame(y = (data$eruptions - mean(data$eruptions)) / sd(data$eruptions))[[1]]

  # act & assert
  expect_error(estimate_model(f1)(data))
  expect_error(estimate_model(f2))
  expect_error(estimate_model(f3)(data))
  expect_error(estimate_model(f4))
})


# ---- inv_transform_response ----
test_that("inv_transform_response yields a function that transforms response vars as expected", {
  # arrange
  data <- faithful
  f <- inv_transform_response(function(df) {
    data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
  })

  # act
  f_out <- f(data)

  # assert
  output <- cbind(data, y = (data$eruptions - mean(data$eruptions)) / sd(data$eruptions))
  expect_true(is.function(f))
  expect_s3_class(f, "inv_transform_response")
  expect_s3_class(f, "ml_pipeline_section")
  expect_equal(f_out, output)
})


test_that("inv_transform_response throws errors with invalid inputs", {
  # arrange
  data <- faithful
  f1 <- function(df1, df2) {
    data.frame(y = (df1$eruptions - mean(df1$eruptions)) / sd(df1$eruptions))
  }
  f2 <- function(df) {
    data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))[[1]]
  }
  f3 <- data.frame(y = (data$eruptions - mean(data$eruptions)) / sd(data$eruptions))[[1]]

  # act & assert
  expect_error(inv_transform_features(f1))
  expect_error(inv_transform_features(f2)(data))
  expect_error(inv_transform_features(f3))
})


# ---- pipeline ----
test_that("pipeline produces machine learning pipelines with all possible stages", {
  # arrange
  data <- faithful

  # act
  lm_pipeline <-
    pipeline(
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

  # assert
  df <- cbind(
    data,
    data.frame(x1 = (data$waiting - mean(data$waiting)) / sd(data$waiting)),
    data.frame(y = (data$eruptions - mean(data$eruptions)) / sd(data$eruptions))
  )
  manual_model <- lm(y ~ 1 + x1, df)
  manual_pred <- cbind(
    df,
    pred_model = predict(manual_model, df),
    pred_eruptions = data.frame(pred_eruptions = predict(manual_model, df) * sd(df$eruptions) + mean(df$eruptions))
  )

  expect_s3_class(lm_pipeline, "ml_pipeline")
  expect_equal(lm_pipeline$inner_model, manual_model)
  expect_true(is.function(lm_pipeline$predict))
  expect_equal(lm_pipeline$predict(data), manual_pred)
  expect_equal(lm_pipeline$predict(data, verbose = FALSE), manual_pred[6])
})


test_that("pipeline produces machine learning pipelines with just feature transformation", {
  # arrange
  data <- faithful

  # act
  lm_pipeline <-
    pipeline(
      data,
      transform_features(function(df) {
        data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))
      }),

      estimate_model(function(df) {
        lm(eruptions ~ 1 + x1, df)
      })
    )

  # assert
  df <- cbind(
    data,
    data.frame(x1 = (data$waiting - mean(data$waiting)) / sd(data$waiting))
  )
  manual_model <- lm(eruptions ~ 1 + x1, df)
  manual_pred <- cbind(
    df,
    pred_model = predict(manual_model, df)
  )

  expect_s3_class(lm_pipeline, "ml_pipeline")
  expect_equal(lm_pipeline$inner_model, manual_model)
  expect_true(is.function(lm_pipeline$predict))
  expect_equal(lm_pipeline$predict(data), manual_pred)
  expect_equal(lm_pipeline$predict(data, verbose = FALSE), manual_pred[4])
})


test_that("pipeline produces machine learning pipelines with just response transformations", {
  # arrange
  data <- faithful

  # act
  lm_pipeline <-
    pipeline(
      data,
      transform_response(function(df) {
        data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
      }),

      estimate_model(function(df) {
        lm(y ~ 1 + waiting, df)
      }),

      inv_transform_response(function(df) {
        data.frame(pred_eruptions = df$pred_model * sd(df$eruptions) + mean(df$eruptions))
      })
    )

  # assert
  df <- cbind(
    data,
    data.frame(y = (data$eruptions - mean(data$eruptions)) / sd(data$eruptions))
  )
  manual_model <- lm(y ~ 1 + waiting, df)
  manual_pred <- cbind(
    df,
    pred_model = predict(manual_model, df),
    pred_eruptions = data.frame(pred_eruptions = predict(manual_model, df) * sd(df$eruptions) + mean(df$eruptions))
  )

  expect_s3_class(lm_pipeline, "ml_pipeline")
  expect_equal(lm_pipeline$inner_model, manual_model)
  expect_true(is.function(lm_pipeline$predict))
  expect_equal(lm_pipeline$predict(data), manual_pred)
  expect_equal(lm_pipeline$predict(data, verbose = FALSE), manual_pred[5])
})


test_that("pipeline throws an error if only one of transform_response and transform_response is set", {
  # arrange
  data <- faithful

  # act & assert
  expect_error(
    pipeline(
      data,
      transform_response(function(df) {
        data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
      }),

      estimate_model(function(df) {
        lm(y ~ 1 + waiting, df)
      })
    )
  )

  expect_error(
    pipeline(
      data,
      estimate_model(function(df) {
        lm(y ~ 1 + waiting, df)
      }),

      inv_transform_response(function(df) {
        data.frame(pred_eruptions = df$pred_model * sd(df$eruptions) + mean(df$eruptions))
      })
    )
  )
})


test_that("pipeline throws error if estimate_model not set", {
  # arrange
  data <- faithful

  # act
  expect_error(
    pipeline(
      data,
      transform_response(function(df) {
        data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
      }),

      inv_transform_response(function(df) {
        data.frame(pred_eruptions = df$pred_model * sd(df$eruptions) + mean(df$eruptions))
      })
    )
  )
})


test_that("pipeline automatically handles unexpected methods", {
  # arrange
  data <- faithful

  # act
  lm_pipeline <-
    pipeline(
      data,
      transform_response(function(df) {
        data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
      }),

      x <- rnorm(100),

      estimate_model(function(df) {
        lm(y ~ 1 + waiting, df)
      }),

      inv_transform_response(function(df) {
        data.frame(pred_eruptions = df$pred_model * sd(df$eruptions) + mean(df$eruptions))
      }),

      function(df) df
    )

  # assert
  df <- cbind(
    data,
    data.frame(y = (data$eruptions - mean(data$eruptions)) / sd(data$eruptions))
  )
  manual_model <- lm(y ~ 1 + waiting, df)
  manual_pred <- cbind(
    df,
    pred_model = predict(manual_model, df),
    pred_eruptions = data.frame(pred_eruptions = predict(manual_model, df) * sd(df$eruptions) + mean(df$eruptions))
  )

  expect_s3_class(lm_pipeline, "ml_pipeline")
  expect_equal(lm_pipeline$inner_model, manual_model)
  expect_true(is.function(lm_pipeline$predict))
  expect_equal(lm_pipeline$predict(data), manual_pred)
  expect_equal(lm_pipeline$predict(data, verbose = FALSE), manual_pred[5])
})


# ---- ml_pipeline_builder ----
test_that("ml_pipeline_builder() OO API produces the same results as pipeline() functional API", {
  # arrange
  data <- faithful

  # act - functional API
  lm_pipeline_func <-
    pipeline(
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

  predictions_func <- lm_pipeline_func$predict(data)

  # act - OO API
  lm_pipeline_OO <- ml_pipline_builder()

  lm_pipeline_OO$transform_features(function(df) {
     data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))
  })

  lm_pipeline_OO$transform_response(function(df) {
     data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
  })

  lm_pipeline_OO$inv_transform_response(function(df) {
    data.frame(pred_eruptions = df$pred_model * sd(df$eruptions) + mean(df$eruptions))
  })

  lm_pipeline_OO$estimate_model(function(df) {
     lm(y ~ 1 + x1, df)
  })

  lm_pipeline_OO$fit(data)
  predictions_OO <- lm_pipeline_OO$predict(data)

  # assert
  expect_equal(lm_pipeline_OO$inner_model(), lm_pipeline_func$inner_model)
  expect_equal(predictions_OO, predictions_func)
  expect_equal(class(lm_pipeline_OO), class(lm_pipeline_func))
})


# ---- predict.lm_pipeline ----
test_that("predict.ml_pipeline generates non-verbose predictions for a pipeline", {
  # arrange
  data <- faithful

  lm_pipeline <-
    pipeline(
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

  # act
  predictions <- predict(lm_pipeline, data)

  # assert
  result <- lm_pipeline$predict(data)[[6]]
  expect_equal(predictions, result)
})


test_that("predict.ml_pipeline generates verbose predictions for a pipeline", {
  # arrange
  data <- faithful

  lm_pipeline <-
    pipeline(
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

  # act
  predictions <- predict(lm_pipeline, data, verbose = TRUE)

  # assert
  result <- lm_pipeline$predict(data)
  expect_equal(predictions, result)
})


test_that("predict.ml_pipeline generates verbose predictions for pipeline with custom var name", {
  # arrange
  data <- faithful

  lm_pipeline <-
    pipeline(
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
        data.frame(pred_eruptions = df$foo * sd(df$eruptions) + mean(df$eruptions))
      })
    )

  # act
  predictions <- predict(lm_pipeline, data, verbose = TRUE, pred_var = "foo")

  # assert
  result <- lm_pipeline$predict(data, pred_var = "foo")
  expect_equal(predictions, result)
})
