# Breast Cancer Data Analysis
# Group 9 Project

# Load required libraries
library(janitor)
library(dplyr)
library(ggplot2)
library(caret)
library(e1071)
library(pROC)
library(corrplot)

# Read and clean data
data_path <- "data.csv"  # Ensure data.csv is in the working directory
BreastCancerData <- read.csv(data_path) %>%
  clean_names() %>%
  distinct() %>%
  select(-starts_with("x")) %>%
  mutate(diagnosis = factor(diagnosis, levels = c("M", "B")))

# Drop unwanted columns
columns_to_drop <- c("concavity_mean", "concave_points_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
                     "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
                     "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
                     "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst")
BreastCancerData <- BreastCancerData[, !names(BreastCancerData) %in% columns_to_drop]

# Create tumor size category
tumor_quartiles <- quantile(BreastCancerData$radius_mean, probs = c(0, 0.25, 0.5, 0.75, 1))
BreastCancerData$tumor_size <- cut(BreastCancerData$radius_mean, 
                                    breaks = tumor_quartiles, 
                                    labels = c("Very Small Tumors", "Small Tumors", "Medium Tumors", "Large Tumors"), 
                                    include.lowest = TRUE)
BreastCancerData$tumor_size_numerical <- as.integer(as.factor(BreastCancerData$tumor_size))

# Normality checks
normality_results <- lapply(BreastCancerData[c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                                              "smoothness_mean", "compactness_mean", "symmetry_mean", "fractal_dimension_mean")], 
                           function(x) shapiro.test(x)$p.value)
print(normality_results)

# Skewness
skewness <- function(x) { mean((x - mean(x))^3) / (sd(x)^3) }
skew_vals <- sapply(BreastCancerData[c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
                                        "smoothness_mean", "compactness_mean", "symmetry_mean", "fractal_dimension_mean")], skewness)
print(skew_vals)

# Outlier Treatment
treat_outliers <- function(x) {
  threshold <- mean(x, na.rm = TRUE) + 3 * sd(x, na.rm = TRUE)
  x[x > threshold] <- NA
  return(x)
}

for (v in names(skew_vals)) {
  BreastCancerData[[v]] <- treat_outliers(BreastCancerData[[v]])
}

# Mann-Whitney U Tests
features <- c("radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "symmetry_mean", "fractal_dimension_mean")
for (feature in features) {
  result <- wilcox.test(as.formula(paste(feature, "~ diagnosis")), data = BreastCancerData)
  print(paste("Mann-Whitney test for", feature, ": p-value =", result$p.value))
}

# Correlation analysis
numeric_features <- BreastCancerData[sapply(BreastCancerData, is.numeric)]
cor_matrix <- cor(numeric_features, use = "complete.obs")
corrplot(cor_matrix, method = "color", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.8)

# Model: SVM
set.seed(123)
split_index <- createDataPartition(BreastCancerData$diagnosis, p = 0.8, list = FALSE)
train_data <- BreastCancerData[split_index, ]
test_data <- BreastCancerData[-split_index, ]

svm_model <- svm(diagnosis ~ ., data = train_data, type = "C-classification", kernel = "radial", probability = TRUE)
svm_predictions <- predict(svm_model, test_data)
svm_accuracy <- mean(svm_predictions == test_data$diagnosis)
print(paste("SVM Accuracy:", svm_accuracy))

# Model: Logistic Regression
log_model <- glm(diagnosis ~ ., data = train_data, family = binomial())
log_probs <- predict(log_model, test_data, type = "response")
log_pred_class <- ifelse(log_probs > 0.5, levels(train_data$diagnosis)[1], levels(train_data$diagnosis)[2])
log_conf <- confusionMatrix(as.factor(log_pred_class), test_data$diagnosis)
print(log_conf)

# ROC Curve for Logistic Regression
roc_obj <- roc(test_data$diagnosis, log_probs)
plot(roc_obj, main = "Logistic Regression ROC")
print(paste("AUC:", auc(roc_obj)))

# Save cleaned dataset
write.csv(BreastCancerData, "cleaned_breast_cancer_data.csv", row.names = FALSE)

