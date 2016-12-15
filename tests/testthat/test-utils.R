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


context('utils')


# ---- predict_model ----
test_that("predict_model yields a function of a data.frame, returned with predictions appended", {
  # arrange
  data <- faithful
  model <- lm(eruptions ~ 1 + waiting, data)

  # act
  df_predict <- predict_model(model)(data)

  # assert
  expect_s3_class(df_predict, "data.frame")
  expect_equal(colnames(df_predict), c(colnames(data), "pred_model"))
  expect_equal(df_predict$pred_model, stats::predict(model, data), check.names = FALSE)
})


test_that("predict_model can rename the default prediction column", {
  # arrange
  data <- faithful
  model <- lm(eruptions ~ 1 + waiting, data)
  new_pred_col_name <- "the_predictions"

  # act
  df_predict <- predict_model(model)(data, new_pred_col_name)
  df_predict_name <- colnames(df_predict)[dim(df_predict)[2]]

  # assert
  expect_equal(df_predict_name, new_pred_col_name)
})


# ---- check_data_frame_throw_error ----
test_that("check_data_frame_throw_error doesn't throw an error when the object is a data.frame", {
  # arrange
  input <- data.frame(x = 1:5, y = 6:10)

  # act & assert
  expect_null(check_data_frame_throw_error(input, "returning_function_name"))
})


test_that('check_data_frame_throw_error throws an error when the object is not a data.frame', {
  # arrange
  input <- 1:10

  # act & assert
  expect_error(check_data_frame_throw_error(input, "returning_function_name"))
})


# ---- process_transform_throw_error ----
test_that("process_transform_throw_error doesn't throw an error for valid transform operations", {
  # arrange
  df_in <- data.frame(y = 1:5, x = 1:5 / 10)
  df_out <- data.frame(q = df_in$x * 2)

  # act
  df_checked <- process_transform_throw_error(df_in, df_out, "hand-written()")

  # assert
  expect_equal(df_checked, df_out)
})


test_that("process_transform_throw_error throws an error if the output is not a data.frame", {
  # arrange
  df_in <- data.frame(y = 1:5, x = 1:5 / 10)
  df_out <- data.frame(q = df_in$x * 2)[[1]]

  # act & assert
  expect_error(process_transform_throw_error(df_in, df_out, "hand-written()"))
})


test_that("process_transform_throw_error removes potentially duplicated columns", {
  # arrange
  df_in <- data.frame(y = 1:5, x = 1:5 / 10)
  df_out <- cbind(df_in, data.frame(q = df_in$x * 2))

  # act
  df_checked <- process_transform_throw_error(df_in, df_out, "hand-written()")

  # assert
  df_unique_cols <- df_out <- data.frame(q = df_in$x * 2)
  expect_equal(df_checked, df_unique_cols)
})


test_that("process_transform_throw_error throws an error if output data.frame is empty", {
  # arrange
  df_in <- data.frame(y = 1:5, x = 1:5 / 10)
  df_out <- df_in

  # act & assert
  expect_error(process_transform_throw_error(df_in, df_out, "hand-written()"))
})


# ---- check_unary_func_throw_error ----
test_that("check_unary_func_throw_error throws an error if a function isn't unary", {
  # arrange
  func <- function(x, y) x + y

  # act & assert
  expect_error(check_unary_func_throw_error(func))
})


test_that("check_unary_func_throw_error doesn't throw an error if a function is unary", {
  # arrange
  func <- function(x) x

  # act & assert
  expect_null(check_unary_func_throw_error(func))
})


test_that("check_unary_func_throw_error doesn't throw an error if arg is not a function", {
  # arrange
  func <- data.frame(x = 1:5)

  # act & assert
  expect_error(check_unary_func_throw_error(func))
})


# ---- check_predict_method_throw_error ----
test_that("check_predict_method_throw_error doesn't throw error if object has predict method", {
  # arrange
  data <- faithful
  model <- lm(eruptions ~ 1 + waiting, data)

  # act & assert
  expect_null(check_predict_method_throw_error(model))
})


test_that("check_predict_method_throw_error throws error if object doesn't have predict method", {
  # arrange
  data <- faithful
  model <- lm(eruptions ~ 1 + waiting, data)
  class(model) <- NULL

  #act & assert
  expect_error(check_predict_method_throw_error(model))
})


# ---- cbind_fast ----
test_that("cbind_fast works like cbind for simple data.frame column combination", {
  # arrange
  df1 <- data.frame(x = 1:5, y = 1:5 * 0.1)
  df2 <- data.frame(a = 6:10, b = 6:10 * 0.25)

  # act & assert

  expect_equal(cbind_fast(df1, df2), cbind(df1, df2))
})
