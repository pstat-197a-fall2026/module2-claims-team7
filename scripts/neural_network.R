
library(tidyverse)
library(rsample)
library(yardstick)
library(keras)
library(tensorflow)

if (!exists("task_results")) source("scripts/tasks-1-2.R")

tensorflow::set_random_seed(110122)

text_col <- "heading_text"

test_pages <- testing(split)
test_text  <- test_pages[[text_col]]

set.seed(110122)
inner <- group_initial_split(train_pages, group = original_url, prop = 0.8)
fit_pages <- training(inner)
val_pages <- testing(inner)

max_tokens <- 5000

build_model <- function(adapt_text, units, dropout, lr) {
  vec_layer <- layer_text_vectorization(
    standardize = NULL, split = "whitespace", ngrams = NULL,
    max_tokens = max_tokens, output_mode = "tf_idf"
  )
  vec_layer %>% adapt(adapt_text)

  model <- keras_model_sequential() %>%
    vec_layer() %>%
    layer_dropout(dropout) %>%
    layer_dense(units = units, activation = "relu") %>%
    layer_dropout(dropout) %>%
    layer_dense(1) %>%
    layer_activation(activation = "sigmoid")

  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = lr),
    metrics = "binary_accuracy"
  )
  model
}

grid <- expand_grid(units = c(16, 64), dropout = c(0.2, 0.5), lr = 1e-3)

tune_results <- pmap_dfr(grid, function(units, dropout, lr) {
  model <- build_model(fit_pages[[text_col]], units, dropout, lr)
  history <- model %>% fit(
    fit_pages[[text_col]], as.numeric(fit_pages$relevant),
    validation_data = list(val_pages[[text_col]], as.numeric(val_pages$relevant)),
    epochs = 30, batch_size = 32, verbose = 0,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 3,
                                             restore_best_weights = TRUE))
  )
  best_epoch <- which.min(history$metrics$val_loss)
  tibble(units, dropout, lr,
         best_epoch,
         val_loss = history$metrics$val_loss[best_epoch],
         val_accuracy = history$metrics$val_binary_accuracy[best_epoch])
})
print(tune_results)

best <- tune_results %>% slice_min(val_loss, n = 1, with_ties = FALSE)

final_model <- build_model(train_pages[[text_col]], best$units, best$dropout, best$lr)
final_model %>% fit(
  train_pages[[text_col]], as.numeric(train_pages$relevant),
  epochs = best$best_epoch, batch_size = 32, verbose = 0
)

nn_prob <- predict(final_model, test_text) %>% as.numeric()
nn_prob <- pmin(pmax(nn_prob, 1e-7), 1 - 1e-7)
nn_row  <- score(qlogis(nn_prob)) %>%
  mutate(model = "Neural network", .before = 1)

all_results <- bind_rows(task_results, nn_row)
print(all_results)


dir.create("results", showWarnings = FALSE)
save_model_tf(final_model, "results/nn-binary-model")
save(all_results, tune_results, file = "results/task-3-results.RData")
