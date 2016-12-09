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


#' Build Machine Learning Pipelines
#'
#' @export
#'
#' @return
#'
#' @examples
ml_pipline_builder <- function() {
  #
  this <- environment()

  #
  model_estimate <- NULL
  pipeline_predict <- NULL

  #
  transform_features <- function(x) NULL
  transform_response <- function(x) NULL
  inv_transform_response <- function(x) NULL
  estimate_model <- function(x) NULL

  #
  pipeline_fit <- function(data) {
    check_data_frame_throw_error(data, "data")

    model_features <- transform_features(data)
    check_data_frame_throw_error(model_features, "transform_features()")

    model_response <- transform_response(data)
    check_data_frame_throw_error(model_features, "transform_response()")

    model_data <- do.call(cbind,
                          Filter(function(x) !is.null(x), list(data, model_features, model_response)))

    model_estimate <<- estimate_model(model_data)

    #
    pipeline_predict <<- function(data) {
      check_data_frame_throw_error(data, "data")

      model_features <- transform_features(data)
      model_data <- do.call(cbind,
                            Filter(function(x) !is.null(x), list(data, model_features)))

      model_predictions <- data.frame("pred" = stats::predict(model_estimate, model_data))
      transformed_predictions <- inv_transform_response(model_predictions)
      check_data_frame_throw_error(transformed_predictions, "inv_transform_response()")

      do.call(cbind, Filter(function(x) !is.null(x), list(model_data, model_predictions,
                                                          transformed_predictions)))
    }
  }

  structure(this, class = c("ml_pipeline"))
}


#' Validate ml_pipeline_builder method returns data.frame
#'
#'
#' Helper function that checks if the object returned from a \code{ml_pipeline_builder method} is
#' data.frame (if it isn't NULL), and if it isn't, throws an error that is customised with the
#' returning function name.
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
    stop(paste(func_name, "transform_response() does not produce a data.frame"))
  }

  NULL
}
