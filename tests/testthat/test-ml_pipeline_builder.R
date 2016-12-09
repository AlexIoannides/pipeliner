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


context('ml_pipeline_builder')


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
