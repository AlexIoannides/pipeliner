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
#' lm_pipeline$transform_features(function(df) {
#'   data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))
#' })
#'
#' lm_pipeline$transform_response(function(df) {
#'   data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
#' })
#'
#' lm_pipeline$inv_transform_response(function(df) {
#'   data.frame(pred_eruptions = df$pred_model * sd(df$eruptions) + mean(df$eruptions))
#' })
#'
#' lm_pipeline$estimate_model(function(df) {
#'   lm(y ~ 0 + x1, df)
#' })
#'
#' lm_pipeline$fit(data)
#' head(lm_pipeline$predict(data))
#' #    eruptions waiting         x1 pred_model pred_eruptions
#' #  1     3.600      79  0.5960248  0.5369058       4.100592
#' #  2     1.800      54 -1.2428901 -1.1196093       2.209893
#' #  3     3.333      74  0.2282418  0.2056028       3.722452
#' #  4     2.283      62 -0.6544374 -0.5895245       2.814917
#' #  5     4.533      85  1.0373644  0.9344694       4.554360
#' #  6     2.883      55 -1.1693335 -1.0533487       2.285521
#'
ml_pipline_builder <- function() {
  # # capture the contents of the local environment
  # this <- environment()

  # define inner fields for estimated models and pipeline prediction functions
  model_estimate_inner <- NULL
  predict_inner <- NULL

  # define inner methods for transformation and estimation that need to be set
  transform_features_inner <- function(df) NULL
  transform_response_inner <- function(df) NULL
  inv_transform_response_inner <- function(df) NULL
  estimate_model_inner <- function(df) NULL

  # define interface for setting transformation and estimation methods
  model_estimate <- function() {
    model_estimate_inner
  }

  predict <- function(data, verbose_data_frame = TRUE, ...) {
    predict_inner(data, verbose_data_frame = TRUE, ...)
  }

  transform_features <- function(f) {
    check_unary_func_throw_error(f, "transform_features()")
    transform_features_inner <<- function(df) {
      new_df <- f(df)
      process_transform_throw_error(df, new_df, "transform_features()")
    }
  }

  transform_response <- function(f) {
    check_unary_func_throw_error(f, "transform_response()")
    transform_response_inner <<- function(df) {
      new_df <- f(df)
      process_transform_throw_error(df, new_df, "transform_response()")    }
  }

  inv_transform_response <- function(f) {
    check_unary_func_throw_error(f, "inv_transform_response()")
    inv_transform_response_inner <<- function(df) {
      new_df <- f(df)
      process_transform_throw_error(df, new_df, "inv_transform_response()")    }
  }

  estimate_model <- function(f) {
    check_unary_func_throw_error(f, "estimate_model()")
    estimate_model_inner <<- function(df) {
      new_model <- f(df)
      check_predict_method_throw_error(new_model)
      new_model
    }
  }

  # method for fitting the pipeline and which creates and sets the model pipeline
  fit <- function(data) {
    check_data_frame_throw_error(data, "data")

    model_features <- transform_features_inner(data)
    model_response <- transform_response_inner(data)

    model_data <- do.call(cbind,
                          Filter(function(x) !is.null(x), list(data, model_features, model_response)))

    model_estimate_inner <<- estimate_model_inner(model_data)

    # assemble and set the pipeline prediction object
    predict_inner <<- function(data, verbose_data_frame = TRUE, ...) {
      check_data_frame_throw_error(data, "data")

      model_features <- transform_features_inner(data)
      model_data <- do.call(cbind,
                            Filter(function(x) !is.null(x), list(data, model_features)))

      model_predictions <- data.frame("pred_model" = stats::predict(model_estimate_inner, model_data, ...))
      model_data <- cbind(model_data, model_predictions)
      transformed_predictions <- inv_transform_response_inner(model_data)

      verbose_output <- do.call(cbind, Filter(function(x) !is.null(x), list(model_data,
        transformed_predictions)))

      if (verbose_data_frame) {
        return(verbose_output)
      } else {
        return(verbose_output[dim(verbose_output)[2]])
      }
    }
  }

  # return the object
  interface <- list("transform_features" = transform_features,
    "transform_response" = transform_response, "inv_transform_response" = inv_transform_response,
    "estimate_model" = estimate_model, "model_estimate" = model_estimate, "predict" = predict,
    "fit" = fit)

  structure(interface, class = c("ml_pipeline"))
}


#' Predict method for ML pipelines
#'
#' @export
#'
#' @param pipeline_object An estimated pipleine object of class \code{ml_pipeline}.
#' @param new_data A data.frame in which to look for variables with which to predict.
#' @param ... Any additional arguements than need to be passed to the underlying model's predict
#' methods.
#'
#' @return A data.frame containing the predicted values, input variables, as well as any interim
#' tranformed variables.
#'
#' @examples
#' lm_pipeline <- ml_pipline_builder()
#' lm_pipeline$estimate_model(function(df) lm(y ~ x, df))
#' data <- data.frame(x = c(-1.26, 1.24, 0.54), y = 0.5 * c(-1.26, 1.24, 0.54) + rnorm(3))
#' lm_pipeline$fit(data)
#' predict(lm_pipeline, data)
#' # x          y       pred
#' # 1 -1.26 -0.6440193 -0.9355810
#' # 2  1.24  1.7833237  1.0335936
#' # 3  0.54 -0.5590672  0.4822247
predict.ml_pipeline <- function(pipeline_object, new_data, ...) {
  pipeline_object$predict(new_data, verbose_data_frame = FALSE, ...)[[1]]
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
#' # transform_method yields data.frame that duplicates input vars - dropping the following columns: 'y', 'x'
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


