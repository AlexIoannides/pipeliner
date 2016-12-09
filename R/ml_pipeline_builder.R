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


#' Build machine learning pipelines
#'
#' Building machine learning models often requires pre- and post-transformation of the input and/or
#' response variables, prior to training (or fitting) the models. For example, a model may require
#' training on the logarithm of the response and input variables. As a consequence, fitting and
#' then generating predictions from these models requires repeated application of transformation and
#' inverse-transormation functions, to go from the original input to original output variables (via
#' the model).
#'
#' This function produces an object in which it is possible to: define transformation and
#' inverse-transformation functions; fit a model on training data; and then generate a prediction
#' (or model-scoring) function that automatically applies the entire pipeline of transformation and
#' inverse-transformation to the inputs and outputs of the inner-model's predicted scores.
#'
#' Calling \code{ml_pipline_builder()} will return an 'ml_pipeline' object (actually an environment
#' or closure), whose methods can be accessed as one would access any element of a list. For example,
#' \code{ml_pipline_builder()$transform_features} will allow you to get or set the
#' \code{transform_features} function to use the pipeline. The full list of methods are:
#'
#' \itemize{
#' \item \code{transform_features} - a unary function of a data.frame that returns a new data.frame
#' containing only the transformed input variables - an error will be thrown if this is not the
#' case (unless the function has been left undefined).
#'
#' \item \code{transform_response} - a unary function of a data.frame that returns a new data.frame
#' containing only the transformed response variables - an error will be thrown if this is not the
#' case (unless the function has been left undefined).
#'
#' \code{inv_transform_response} - this is the inverse of the \code{transform_response} function, a
#' unary function of a data.frame that takes raw model output and transforms it back into the space
#' containing the original data, returning a data.frame containing only this variable. An error
#' will be thrown if any of these criteria are not met (unless the function has been left
#' undefined).
#'
#' \item \code{estimate_model} - a unary function of a data.frame that returns a fitted model
#' object, which must have a \code{predict.{model-class}} defined and available in the enclosing
#' environment. An error will be thrown if any of these criteria are not met.
#'
#' \item \code{pipeline_fit} - a unary function of the input data that will apply (if defined)
#' \code{transform_features} and \code{transform_response}, before executing \code{estimate_model} to
#' estimate the model and create the end-to-end model pipeline.
#'
#' \item \code{pipeline_predict} - returns the model prediction pipeline (otherwise NULL).
#'
#' \item \code{model_estimate} - returns the estimated inner model object used in the pipeline
#' (otherwise NULL).
#' }
#'
#' @export
#'
#' @return An object of class \code{ml_pipeline}.
#'
#' @examples
#' data <- faithful
#'
#' lm_pipeline <- ml_pipline_builder()
#'
#' lm_pipeline$transform_features <- function(df) {
#'   data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))
#' }
#'
#' lm_pipeline$transform_response <- function(df) {
#'   data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
#' }
#'
#' lm_pipeline$inv_transform_response <- function(df) {
#'   data.frame(pred_eruptions = df$pred * sd(df$eruptions) + mean(df$eruptions))
#' }
#'
#' lm_pipeline$estimate_model <- function(df) {
#'   lm(y ~ 0 + x1, df)
#' }
#'
#' lm_pipeline$pipeline_fit(data)
#'
#' summary(lm_pipeline$model_estimate)
#'
#' in_sample_predictions <- lm_pipeline$pipeline_predict(data)
#' head(in_sample_predictions)
#' #   eruptions waiting         x1       pred pred_eruptions
#' # 1     3.600      79  0.5960248  0.5369058       4.100592
#' # 2     1.800      54 -1.2428901 -1.1196093       2.209893
#' # 3     3.333      74  0.2282418  0.2056028       3.722452
#' # 4     2.283      62 -0.6544374 -0.5895245       2.814917
#' # 5     4.533      85  1.0373644  0.9344694       4.554360
#' # 6     2.883      55 -1.1693335 -1.0533487       2.285521
ml_pipline_builder <- function() {
  # capture the contents of the local environment
  this <- environment()

  # define fields for estimated models and pipeline prediction functions
  model_estimate <- NULL
  pipeline_predict <- NULL

  # define methods for transformation and estimation that need to be set
  transform_features <- function(x) NULL
  transform_response <- function(x) NULL
  inv_transform_response <- function(x) NULL
  estimate_model <- function(x) NULL

  # method for fitting the pipeline and which creates and sets the model pipeline
  pipeline_fit <- function(data) {
    check_data_frame_throw_error(data, "data")

    model_features <- transform_features(data)
    check_data_frame_throw_error(model_features, "transform_features()")

    model_response <- transform_response(data)
    check_data_frame_throw_error(model_features, "transform_response()")

    model_data <- do.call(cbind,
                          Filter(function(x) !is.null(x), list(data, model_features, model_response)))

    model_estimate <<- estimate_model(model_data)

    # assemble and set the pipeline prediction object
    pipeline_predict <<- function(data) {
      check_data_frame_throw_error(data, "data")

      model_features <- transform_features(data)
      model_data <- do.call(cbind,
                            Filter(function(x) !is.null(x), list(data, model_features)))

      model_predictions <- data.frame("pred" = stats::predict(model_estimate, model_data))
      model_data <- cbind(model_data, model_predictions)
      transformed_predictions <- inv_transform_response(model_data)
      check_data_frame_throw_error(transformed_predictions, "inv_transform_response()")

      do.call(cbind, Filter(function(x) !is.null(x), list(model_data, transformed_predictions)))
    }
  }

  # return the object
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
