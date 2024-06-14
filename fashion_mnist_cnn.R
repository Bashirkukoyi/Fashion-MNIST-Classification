install.packages("keras")
library(keras)
install_keras()

library(keras)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Normalize the images
train_images <- train_images / 255
test_images <- test_images / 255

# Reshape the data
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Define the class names
class_names <- c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# Build the model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Train the model
model %>% fit(train_images, train_labels, epochs = 10, validation_data = list(test_images, test_labels))

# Evaluate the model
model %>% evaluate(test_images, test_labels)

# Make predictions
predictions <- model %>% predict(test_images)

# Plot some predictions
plot_image <- function(i, predictions_array, true_label, img) {
  true_label <- true_label[i]
  img <- img[i,,]
  
  plot(as.raster(img, max = 1), main = paste(class_names[which.max(predictions_array) + 1], 
                                             sprintf("%.2f%%", 100*max(predictions_array)), 
                                             "\n(True:", class_names[true_label + 1], ")"))
}

par(mfrow=c(5,3))
for (i in 1:15) {
  plot_image(i, predictions[i,], test_labels, test_images)
}
