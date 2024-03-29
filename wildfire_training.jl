# Import necessary libraries
using Flux
using BSON
using Random
using LinearAlgebra
using Flux: mse, ADAM

training_data = BSON.load("wildfire_training_data_1.bson")
global keys_of_dictionary = keys(training_data)
global input_data = []
global target_data = []

# Extract input and target data from the dictionary
for i in keys_of_dictionary
    if i != "load_shed"
        push!(input_data, training_data["$i"])
    end
end

input_data = hcat(input_data...)
# target_data = hcat(target_data...)
target_data = hcat(training_data["load_shed"])

function train_test_split(data, test_ratio=0.2, seed=42)
    Random.seed!(seed)  # Set a seed for reproducibility
    indices = randperm(size(data, 1))
    n_test = Int(floor(test_ratio * length(indices)))
    return data[indices[1:end-n_test], :], data[indices[end-n_test+1:end], :]
end

# Split the data into training nd testing datasets
x_train, x_test = train_test_split(input_data, 0.2, 42)
y_train, y_test = train_test_split(target_data, 0.2, 42)
x_train = x_train'
y_train = y_train'
x_test = x_test'
y_test = y_test'
fd
# Define the model parameters
input_layer_size = size(x_train)[1]
output_layer_size = size(y_train)[1]
model = Chain(
        Dense(input_layer_size, 560, relu),
        Dense(560, 160, relu),
        Dense(160, 30, relu),
        Dense(30, output_layer_size)
)


# Hyperparameters
epochs = 1000
learning_rate = 0.001
patience = 20  # Number of epochs to wait before early stopping if no improvement

# Train the neural network
loss(x, y) = Flux.mae(model(x), y)
opt = ADAM(learning_rate)
data = [(x_test, y_test)]

global best_loss = Inf
global wait = 0

for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), data, opt)
    val_loss = loss(x_test, y_test)
    println("Epoch $epoch, Loss: $val_loss")

    # Early stopping
    if epoch == 1 || val_loss < best_loss
        global best_loss = val_loss
        global wait = 0
    else
        global wait += 1
        if wait >= patience
            println("Early stopping at epoch $epoch")
            break
        end
    end
end

predictions = model(x_test)
errors = predictions - y_test
relative_error = norm(errors)/norm(y_test) * 100
println("Relative error of prediction: $relative_error")

# Function to save the trained model to a BSON file
function save_model_to_bson(filename, model)
    BSON.@save filename model=model
end
# Save the trained model to a BSON file
model_filename = "wildfire_trained_model.bson"
save_model_to_bson(model_filename, model)
println("Training and saving completed.")
