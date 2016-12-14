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


#' Generate machine learning model prediction
#'
#' A helper function that takes as its arguement an estimated machine learning model and returns a
#' prediction function for use within a machine learning pipeline.
#'
#' @param .m An estimated machine lerning model.
#'
#' @return A unary function of a data.frame that returns the input data.frame with the predicted
#' response variable column appended. This function is assigned the classes
#' \code{"predict_model"} and \code{"ml_pipeline_section"}.
#'
#' @examples
#' \dontrun{
#' data <- head(faithful)
#' m <- estimate_model(function(df) {
#'   lm(eruptions ~ 1 + waiting, df)
#' })
#'
#' predict_model(m(data))(data, "pred_eruptions")
#' #   eruptions waiting pred_eruptions
#' # 1     3.600      79       3.803874
#' # 2     1.800      54       2.114934
#' # 3     3.333      74       3.466086
#' # 4     2.283      62       2.655395
#' # 5     4.533      85       4.209219
#' # 6     2.883      55       2.182492
#' }
predict_model <- function(.m) {
  g <- function(df_in, pred_var = "pred_model", ...) {
    check_data_frame_throw_error(df_in)
    df_out <- stats::setNames(data.frame(stats::predict(.m, df_in, ...)), pred_var)
    cbind(df_in, df_out)
  }

  structure(g, class = c("predict_model", "ml_pipeline_section"))
}


#' Validate ml_pipeline_builder transform method returns data.frame
#'
#'
#' Helper function that checks if the object returned from a \code{ml_pipeline_builder} method is
#' data.frame (if it isn't NULL), and if it isn't, throws an error that is customised with the
#' returning name.
#'
#' @param func_return_object The object returned from a \code{ml_pipeline_builder} method.
#' @param func_name The name of the function that returned the object.
#'
#' @return NULL
#'
#' @examples
#' \dontrun{
#' transform_method <- function(df) df
#' data <- data.frame(y = c(1, 2), x = c(0.1, 0.2))
#' data_transformed <- transform_method(data)
#' check_data_frame_throw_error(data_transformed, "transform_method")
#' # NULL
#' }
check_data_frame_throw_error <- function(func_return_object, func_name) {
  if (!is.null(func_return_object) & !is.data.frame(func_return_object)) {
    stop(paste(func_name, "does not produce a data.frame."), call. = FALSE)
  }

  NULL
}


#' Validate and clean transform function output
#'
#' Helper function that ensures the output of applying a transform function is a data.frame and
#' that this data frame does not duplicate variables from the original (input data) data frame. If
#' duplicates are found they are automatically dropped from the data.frame that is returned by this
#' function.
#'
#' @param input_df The original (input data) data.frame - the transform function's argument.
#' @param output_df The the transform function's output.
#' @param func_name The name of the \code{ml_pipeline_builder} trandform method.
#'
#' @return If the transform function is not \code{NULL} then a copy of the transform function's
#' output data.frame, with any duplicated inputs removed.
#'
#' @examples
#' \dontrun{
#' transform_method <- function(df) cbind(df, q = df$y * df$y)
#' data <- data.frame(y = c(1, 2), x = c(0.1, 0.2))
#' data_transformed <- transform_method(data)
#' process_transform_throw_error(data, data_transformed, "transform_method")
#' # transform_method yields data.frame that duplicates input vars - dropping the following
#' columns: 'y', 'x'
#' # q
#' # 1 1
#' # 2 4
#' }
process_transform_throw_error <- function(input_df, output_df, func_name) {
  input_vars <- colnames(input_df)
  if (!is.data.frame(output_df)) {
    stop(paste(func_name, "does not produce a data.frame."), call. = FALSE)
  } else {
    output_vars <- colnames(output_df)
    input_vars_in_output_vars <- output_vars %in% input_vars
    if (any(input_vars_in_output_vars)) {
      duplicated_vars <- output_vars[input_vars_in_output_vars]
      output_df[, duplicated_vars] <- NULL
      message_string <- paste(func_name,
                              "yields data.frame that duplicates input vars - dropping the following columns:",
                              paste0(paste0("'", duplicated_vars, "'"), collapse = ", "))
      message(message_string)
    }
  }

  if (dim(output_df)[2] == 0) {
    stop(paste(func_name, "does not produce a data.frame with at least 1 column."), call. = FALSE)
  }

  output_df
}


#' Validate ml_pipeline_builder transform method is a unary function
#'
#'
#' Helper function that checks if a \code{ml_pipeline_builder} method is unary function (if it
#' isn't a NULL returning function), and if it isn't, throws an error that is customised with the
#' method function name.
#'
#' @param func A \code{ml_pipeline_builder} method.
#' @param func_name The name of the \code{ml_pipeline_builder} method.
#'
#' @return NULL
#'
#' @examples
#' \dontrun{
#' transform_method <- function(df) df
#' check_unary_func_throw_error(transform_method, "transform_method")
#' # NULL
#' }
check_unary_func_throw_error <- function(func, func_name) {
  if (!is.null(body(func)) & !(length(formals(func)) == 1)) {
    stop(paste(func_name, "is not a unary function."), call. = FALSE)
  }

  NULL
}


#' Validate estimate_model method returns an object with a \code{predict} method defined
#'
#'
#' Helper function that checks if the object returned from the \code{estimate_model} method has
#' a \code{predict} method defined for it.
#'
#' @param func_return_object The object returned from the \code{estimate_model} method.
#'
#' @return NULL
#'
#' @examples
#' \dontrun{
#' estimation_method <- function(df) lm(eruptions ~ 0 + waiting, df)
#' data <- faithful
#' model_estimate <- estimation_method(data)
#' check_predict_method_throw_error(model_estimate)
#' # NULL
#' }
check_predict_method_throw_error <- function(func_return_object) {
  if (!is.null(func_return_object)) {
    func_return_object_classes <- class(func_return_object)
    has_predict_method <- any(sapply(paste0("predict.", func_return_object_classes), exists))
    if (!has_predict_method) {
      stop("estimate_model() method does not return an object with a predict.{class-name} method.",
           call. = FALSE)
    }
  }

  NULL
}