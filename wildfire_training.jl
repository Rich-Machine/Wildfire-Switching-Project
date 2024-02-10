# Import necessary libraries
using Flux
using BSON
using Random
using Flux: mse, ADAM

training_data = BSON.load("wildfire_training_data.bson")
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
xx
function train_test_split(data, test_ratio=0.2, seed=42)
    Random.seed!(seed)  # Set a seed for reproducibility
    indices = randperm(size(data, 1))
    n_test = Int(floor(test_ratio * length(indices)))
    return data[indices[1:end-n_test], :], data[indices[end-n_test+1:end], :]
end

# Split the data into training nd testing datasets
x_train, x_test = train_test_split(input_data, 0.8, 42)
y_train, y_test = train_test_split(target_data, 0.8, 42)
x_train = x_train'
y_train = y_train'
x_test = x_test'
y_test = y_test'

# Define the model parameters
input_layer_size = size(x_train)[1]
output_layer_size = size(y_train)[1]
model = Chain(
        Dense(input_layer_size, 120, relu),
        Dense(120, 420, relu),
        Dense(420, 220, relu),
        Dense(220, output_layer_size)
        )


# Hyperparameters
epochs = 1000
learning_rate = 0.001

# Train the neural network
loss(x, y) = Flux.mse(model(x), y)

opt = ADAM(learning_rate)
data = [(x_train, y_train)]

for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), data, opt)
    println("Epoch $epoch, Loss: $(loss(x_test, y_test))")
end

# Function to save the trained model to a BSON file
function save_model_to_bson(filename, model)
    BSON.@save filename model=model
end
# Save the trained model to a BSON file
model_filename = "wildfire_trained_model.bson"
save_model_to_bson(model_filename, model)
println("Training and saving completed.")
