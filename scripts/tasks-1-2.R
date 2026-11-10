library(Matrix)
library(rsample)
library(sparsesvd)
library(yardstick)
source("scripts/preprocessing.R")

# extract text and split the labeled pages
load("data/claims-raw.RData")
paragraphs <- parse_data(claims_raw)
headings <- parse_data(claims_raw, selector = "p, h1, h2, h3, h4, h5, h6")

pages <- paragraphs %>%
  transmute(.id, original_url, relevant = bclass == "Relevant claim content",
            paragraph_text = text_clean) %>%
  inner_join(select(headings, .id, heading_text = text_clean), by = ".id") %>%
  mutate(page_row = row_number())

# Keep repeated URLs together and use the same holdout for all comparisons.
set.seed(110122)
split <- group_initial_split(pages, group = original_url, prop = 0.8)
train_pages <- training(split)
train_rows <- train_pages$page_row
test_rows <- testing(split)$page_row

# tokenize text, compute TF-IDF and project onto PCs
# Use the lab's word preprocessing. Keep stopwords in bigrams so each
# token contains two adjacent words from the cleaned text.
make_counts <- function(text, token_type) {
  documents <- tibble(page_row = pages$page_row, text)
  if (token_type == "words") {
    stop_words_clean <- str_remove_all(stop_words$word, "[[:punct:]]")
    tokens <- documents %>%
      unnest_tokens(term, text, token = "words", stopwords = stop_words_clean) %>%
      mutate(term = lemmatize_words(term)) %>%
      filter(str_length(term) > 2)
  } else {
    tokens <- documents %>%
      unnest_tokens(term, text, token = "ngrams", n = 2) %>%
      filter(!is.na(term))
  }

  counts <- count(tokens, page_row, term)
  terms <- unique(counts$term)
  sparseMatrix(i = counts$page_row, j = match(counts$term, terms), x = counts$n,
               dims = c(nrow(documents), length(terms)))
}

# Learn the vocabulary, IDF and sparse SVD projection on training pages.
# A page with no usable text stays in the comparison as a row of zeros.
pc_scores <- function(counts, fit_rows, assessment_rows, n_pc) {
  document_frequency <- colSums(counts[fit_rows, , drop = FALSE] > 0)
  vocabulary <- document_frequency >= 5
  idf <- log(length(fit_rows) / document_frequency[vocabulary])
  tf <- Diagonal(x = 1 / pmax(rowSums(counts), 1)) %*% counts[, vocabulary]
  tf_idf <- tf %*% Diagonal(x = idf)

  fit_matrix <- tf_idf[fit_rows, , drop = FALSE]
  rotation <- sparsesvd(fit_matrix, rank = n_pc)$v
  project <- function(x) {
    scores <- as.data.frame(as.matrix(x %*% rotation))
    names(scores) <- paste0("pc", seq_len(ncol(scores)))
    scores
  }

  list(train = project(fit_matrix),
       assessment = project(tf_idf[assessment_rows, , drop = FALSE]))
}

# Fix the number of components so both word models have the same dimension.
word_pcs <- 50
bigram_pcs <- 20

word_log_odds <- function(counts, fit_rows, assessment_rows) {
  scores <- pc_scores(counts, fit_rows, assessment_rows, word_pcs)
  model <- glm(relevant ~ ., data = data.frame(relevant = pages$relevant[fit_rows],
                                                scores$train),
               family = binomial())
  predict(model, newdata = scores$assessment, type = "link")
}

# Task 1: compare paragraphs alone with paragraphs plus headings 
paragraph_counts <- make_counts(pages$paragraph_text, "words")
heading_counts <- make_counts(pages$heading_text, "words")
paragraph_log_odds <- word_log_odds(paragraph_counts, train_rows, test_rows)
heading_log_odds <- word_log_odds(heading_counts, train_rows, test_rows)

# Task 2: test whether bigrams improve the word model's predictions
# Get out-of-fold word predictions for training the second logistic model.
word_folds <- group_vfold_cv(train_pages, group = original_url, v = 5,
                            balance = "observations")
train_log_odds <- rep(NA_real_, nrow(pages))
for (fold in word_folds$splits) {
  fold_train <- analysis(fold)$page_row
  fold_assessment <- assessment(fold)$page_row
  train_log_odds[fold_assessment] <- word_log_odds(heading_counts, fold_train,
                                                  fold_assessment)
}

bigram_counts <- make_counts(pages$heading_text, "ngrams")
bigram_scores <- pc_scores(bigram_counts, train_rows, test_rows, bigram_pcs)
second_stage_train <- data.frame(relevant = pages$relevant[train_rows],
                                 word_log_odds = train_log_odds[train_rows])
second_stage_test <- data.frame(word_log_odds = heading_log_odds)

# Give the word-only comparison the same opportunity to recalibrate.
word_only_model <- glm(relevant ~ word_log_odds, data = second_stage_train,
                       family = binomial())
word_only_log_odds <- predict(word_only_model, newdata = second_stage_test,
                              type = "link")
bigram_model <- glm(
  relevant ~ .,
  data = cbind(second_stage_train, bigram_scores$train),
  family = binomial()
)
bigram_log_odds <- predict(
  bigram_model,
  newdata = cbind(second_stage_test, bigram_scores$assessment),
  type = "link"
)

# Results for Tasks 1 and 2: compare accuracy and AUC on the same holdout
# Classify pages at a 0.5 probability threshold when computing accuracy.
test_relevant <- pages$relevant[test_rows]
score <- function(log_odds) {
  predicted <- log_odds > 0
  tibble(
    correct = sum(predicted == test_relevant),
    accuracy = mean(predicted == test_relevant),
    auc = roc_auc_vec(factor(test_relevant), log_odds, event_level = "second")
  )
}

task_results <- bind_rows(
  # Task 1: compare HTML extraction strategies.
  "Paragraphs only" = score(paragraph_log_odds),
  "Paragraphs and headings" = score(heading_log_odds),
  # Task 2: compare the two second-stage logistic models.
  "Second stage: word log-odds only" = score(word_only_log_odds),
  "Second stage: word log-odds + bigrams" = score(bigram_log_odds),
  .id = "model"
)
print(task_results)
