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


#' Transform machine learning feature variables
#'
#' A function that takes as its arguement another function defining a set of feature variable
#' transformations, and wraps (or adapts) it for use within a machine learning pipeline.
#'
#' @export
#'
#' @param .f A unary function of a data.frame that returns a new data.frame containing only the
#' transformed feature variables. An error will be thrown if this is not the case.
#'
#' @return A unary function of a data.frame that returns the input data.frame with the transformed
#' feature variable columns appended. This function is assigned the classes
#' \code{"transform_features"} and \code{"ml_pipeline_section"}.
#'
#' @examples
#' data <- head(faithful)
#' f <- transform_features(function(df) {
#'   data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))
#' })
#'
#' f(data)
#' #    eruptions waiting         x1
#' #  1     3.600      79  0.8324308
#' #  2     1.800      54 -1.0885633
#' #  3     3.333      74  0.4482320
#' #  4     2.283      62 -0.4738452
#' #  5     4.533      85  1.2934694
#' #  6     2.883      55 -1.0117236
transform_features <- function(.f) {
  check_unary_func_throw_error(.f, "transform_features()")
  g <- function(df_in) {
    check_data_frame_throw_error(df_in)
    df_out <- process_transform_throw_error(df_in, .f(df_in), "transform_features()")
    cbind(df_in, df_out)
  }

  structure(g, class = c("transform_features", "ml_pipeline_section"))
}


#' Transform machine learning response variable
#'
#' A function that takes as its arguement another function defining a response variable
#' transformation, and wraps (or adapts) it for use within a machine learning pipeline.
#'
#' @export
#'
#' @param .f A unary function of a data.frame that returns a new data.frame containing only the
#' transformed response variable. An error will be thrown if this is not the case.
#'
#' @return A unary function of a data.frame that returns the input data.frame with the transformed
#' response variable column appended. This function is assigned the classes
#' \code{"transform_response"} and \code{"ml_pipeline_section"}.
#'
#' @examples
#' data <- head(faithful)
#' f <- transform_response(function(df) {
#'   data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
#' })
#'
#' f(data)
#' #   eruptions waiting         y
#' # 1     3.600      79  0.5412808
#' # 2     1.800      54 -1.3039946
#' # 3     3.333      74  0.2675649
#' # 4     2.283      62 -0.8088457
#' # 5     4.533      85  1.4977485
#' # 6     2.883      55 -0.1937539
transform_response <- function(.f) {
  check_unary_func_throw_error(.f, "transform_response()")
  g <- function(df_in) {
    check_data_frame_throw_error(df_in)
    df_out <- process_transform_throw_error(df_in, .f(df_in), "transform_response()")
    cbind(df_in, df_out)
  }

  structure(g, class = c("transform_response", "ml_pipeline_section"))
}


#' Estimate machine learning model
#'
#' A function that takes as its arguement another function defining how a machine learning model
#' should be estimated based on the variables available in the input data frame. This function is
#' wrapped (or adapted) for use within a machine learning pipeline.
#'
#' @export
#'
#' @param .f A unary function of a data.frame that returns a fitted model object, which must have
#' a \code{predict.{model-class}} defined and available in the enclosing environment. An error will
#' be thrown if any of these criteria are not met.
#'
#' @return A unary function of a data.frame that returns a fitted model object that has a
#' \code{predict.{model-class}} defined This function is assigned the classes
#' \code{"estimate_model"} and \code{"ml_pipeline_section"}.
#'
#' @examples
#' data <- head(faithful)
#' f <- estimate_model(function(df) {
#'   lm(eruptions ~ 1 + waiting, df)
#' })
#'
#' f(data)
#' # Call:
#' #   lm(formula = eruptions ~ 1 + waiting, data = df)
#' #
#' # Coefficients:
#' # (Intercept)      waiting
#' #    -1.53317      0.06756
estimate_model <- function(.f) {
  check_unary_func_throw_error(.f, "estimate_model()")
  g <- function(df_in) {
    check_data_frame_throw_error(df_in)
    model_out <- .f(df_in)
    check_predict_method_throw_error(model_out)
    model_out
  }

  structure(g, class = c("estimate_model", "ml_pipeline_section"))
}


#' Inverse transform machine learning response variable
#'
#' A function that takes as its arguement another function defining a inverse response variable
#' transformation, and wraps (or adapts) it for use within a machine learning pipeline.
#'
#' @export
#'
#' @param .f A unary function of a data.frame that returns a new data.frame containing only the
#' inverse transformed response variable. An error will be thrown if this is not the case.
#'
#' @return A unary function of a data.frame that returns the input data.frame with the inverse
#' transformed response variable column appended. This function is assigned the classes
#' \code{"inv_transform_response"} and \code{"ml_pipeline_section"}.
#'
#' @examples
#' data <- head(faithful)
#' f1 <- transform_response(function(df) {
#'   data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
#' })
#' f2 <- inv_transform_response(function(df) {
#'   data.frame(eruptions2 = df$y * sd(df$eruptions) + mean(df$eruptions))
#' })
#'
#' f2(f1(data))
#' #   eruptions waiting          y eruptions2
#' # 1     3.600      79  0.5412808      3.600
#' # 2     1.800      54 -1.3039946      1.800
#' # 3     3.333      74  0.2675649      3.333
#' # 4     2.283      62 -0.8088457      2.283
#' # 5     4.533      85  1.4977485      4.533
#' # 6     2.883      55 -0.1937539      2.883
inv_transform_response <- function(.f) {
  check_unary_func_throw_error(.f, "inv_transform_response()")
  g <- function(df_in) {
    check_data_frame_throw_error(df_in)
    df_out <- process_transform_throw_error(df_in, .f(df_in), "inv_transform_response()")
    cbind(df_in, df_out)
  }

  structure(g, class = c("inv_transform_response", "ml_pipeline_section"))
}


#' Build machine learning pipelines - functional API
#'
#' Building machine learning models often requires pre- and post-transformation of the input and/or
#' response variables, prior to training (or fitting) the models. For example, a model may require
#' training on the logarithm of the response and input variables. As a consequence, fitting and
#' then generating predictions from these models requires repeated application of transformation and
#' inverse-transormation functions, to go from the original input to original output variables (via
#' the model).
#'
#' This function that takes individual pipeline sections - functions with class
#' \code{"ml_pipeline_section"} - together with the data required to estimate the inner models,
#' returning a machine pipeline capable of predicting (scoring) data end-to-end, without having to
#' repeatedly apply input variable (feature and response) transformation and their inverses.
#'
#' @export
#'
#' @param .data A data.frame containing the input variables required to fit the pipeline.
#' @param ... Functions of class \code{"ml_pipeline_section"} - e.g. \code{transform_features()},
#' \code{transform_response()}, \code{inv_transform_response()} or \code{estimate_model()}.
#'
#' @return A \code{"ml_pipeline"} object contaiing the pipeline prediction function
#' \code{ml_pipeline$predict()} and the estimated machine learning model nested within it
#' \code{ml_pipeline$inner_model()}.
#'
#' @examples
#' data <- faithful
#'
#' lm_pipeline <-
#'   pipeline(
#'     data,
#'     transform_features(function(df) {
#'       data.frame(x1 = (df$waiting - mean(df$waiting)) / sd(df$waiting))
#'     }),
#'
#'     transform_response(function(df) {
#'       data.frame(y = (df$eruptions - mean(df$eruptions)) / sd(df$eruptions))
#'     }),
#'
#'     estimate_model(function(df) {
#'       lm(y ~ 1 + x1, df)
#'     }),
#'
#'     inv_transform_response(function(df) {
#'       data.frame(pred_eruptions = df$pred_model * sd(df$eruptions) + mean(df$eruptions))
#'     })
#'   )
pipeline <- function(.data, ...) {
  check_data_frame_throw_error(.data)

  # analyse defined pipline sections
  args <- list(...)
  pipes <- Filter(function(x) "ml_pipeline_section" %in% class(x), args)
  pipe_classes <- Map(function(x) class(x)[!(class(x) %in% "ml_pipeline_section")], pipes)

  if (length(unique(pipe_classes)) != length(pipe_classes)) {
    stop("multiple pipeline sections of the same type found - remove from pipe.", call. = FALSE)
  } else {
    names(pipes) <- pipe_classes
  }

  if (is.null(pipes$estimate_model)) stop("estimate_model() undefined", call. = FALSE)

  if (!is.null(pipes$transform_response) & is.null(pipes$inv_transform_response) |
      is.null(pipes$transform_response) & !is.null(pipes$inv_transform_response))  {
    stop("tranform_response() and inv_transform_response() not defined as a pair", call. = FALSE)
  }

  null_pipeline <- Map(function(x) { function(df) df }, vector(mode = "list", length = 4))
  names(null_pipeline) <- c("transform_features", "transform_response", "estimate_model",
                            "inv_transform_response")

  full_pipeline <- Map(function(x) if (is.null(pipes[[x]])) null_pipeline[[x]] else pipes[[x]],
                       names(null_pipeline))

  # estimate model and build prediction pipeline
  inner_model <- full_pipeline$estimate_model(
    full_pipeline$transform_response(
      full_pipeline$transform_features(.data)
    )
  )

  predict_pipeline <- function(df, verbose = TRUE, pred_var = "pred_model", ...) {
    verbose_output <-
      full_pipeline$inv_transform_response(
        predict_model(inner_model)(
          full_pipeline$transform_response(
            full_pipeline$transform_features(df)
          ),
          pred_var,
          ...
        )
      )

    if (verbose) {
      return(verbose_output)
    } else {
      return(verbose_output[dim(verbose_output)[2]])
    }
  }

  # return pipeline object
  interface <- list("predict" = predict_pipeline, "inner_model" = inner_model)
  structure(interface, class = "ml_pipeline")
}


#' Build machine learning pipelines - object oriented API
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
#' \code{transform_features} function to use the pipeline. The full list of methods for defining
#' sections of the pipeline (documented elsewhere) are:
#' \itemize{
#' \item \code{transform_features};
#' \item \code{transform_response};
#' \item \code{inv_transform_response}; and,
#' \item \code{estimate_model};
#' }
#'
#' The pipeline can be fit, prediction generated and the inner model accessed using the following
#' methods:
#' \itemize{
#' \item \code{fit(.data)};
#' \item \code{predict(.data)}; and,
#' \item \code{model_estimate()}.
#' }
#'
#' @export
#'
#' @return An object of class \code{ml_pipeline}.
#'
#' @seealso \code{\link{transform_features}}, \code{\link{transform_response}},
#' \code{\link{estimate_model}} and \code{\link{inv_transform_response}}.
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
  # define inner fields for estimated models and pipeline prediction functions
  inner_model_inner <- NULL
  predict_inner <- NULL

  # define inner methods for transformation and estimation that need to be set
  transform_features_inner <- NULL
  transform_response_inner <- NULL
  inv_transform_response_inner <- NULL
  estimate_model_inner <- NULL

  # define interface for setting transformation and estimation methods
  inner_model <- function() {
    inner_model_inner
  }

  predict <- function(data, verbose = TRUE, pred_var = "pred_model", ...) {
    predict_inner(data, verbose = TRUE, pred_var, ...)
  }

  transform_features <- function(f) {
    transform_features_inner <<- get("transform_features", envir = parent.frame())(f)
  }

  transform_response <- function(f) {
    transform_response_inner <<- get("transform_response", envir = parent.frame())(f)
  }

  inv_transform_response <- function(f) {
    inv_transform_response_inner <<- get("inv_transform_response", envir = parent.frame())(f)
  }

  estimate_model <- function(f) {
    estimate_model_inner <<- get("estimate_model", envir = parent.frame())(f)
  }

  # method for fitting the pipeline and which creates and sets the model pipeline
  fit <- function(data) {
    pipeline_object <- pipeline(data, transform_features_inner, transform_response_inner,
                                inv_transform_response_inner, estimate_model_inner)

    inner_model_inner <<- pipeline_object$inner_model
    predict_inner <<- pipeline_object$predict
  }

  # return the object
  interface <- list("transform_features" = transform_features,
    "transform_response" = transform_response, "inv_transform_response" = inv_transform_response,
    "estimate_model" = estimate_model, "inner_model" = inner_model, "predict" = predict,
    "fit" = fit)

  structure(interface, class = c("ml_pipeline"))
}


#' Predict method for ML pipelines
#'
#' @export
#'
#' @param object An estimated pipleine object of class \code{ml_pipeline}.
#' @param data A data.frame in which to look for input variables with which to predict.
#' @param verbose Boolean - whether or not to return data.frame with all input and interim
#' variables as well as predictions.
#' @param pred_var Name to assign to for column of predictions from the 'raw' (or inner) model in
#' the pipeline.
#' @param ... Any additional arguements than need to be passed to the underlying model's predict
#' methods.
#'
#' @return A vector of model predictions or scores (default); or, a data.frame containing the
#' predicted values, input variables, as well as any interim tranformed variables.
#'
#' @examples
#' data <- faithful
#'
#' lm_pipeline <-
#'   pipeline(
#'     data,
#'     estimate_model(function(df) {
#'       lm(eruptions ~ 1 + waiting, df)
#'     })
#'   )
#'
#' in_sample_predictions <- predict(lm_pipeline, data)
#' head(in_sample_predictions)
#' # [1] 4.100592 2.209893 3.722452 2.814917 4.554360 2.285521
predict.ml_pipeline <- function(object, data, verbose = FALSE, pred_var = "pred_model", ...) {
  if (is.null(object$predict)) {
    stop("predict method not available - check estimation has successfully estimated a model.",
         call. = FALSE)
  }

  if (verbose) {
    object$predict(data, verbose, pred_var, ...)
  } else {
    object$predict(data, verbose, pred_var, ...)[[1]]
  }
}
