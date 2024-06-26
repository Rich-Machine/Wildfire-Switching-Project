# Import necessary libraries
using Flux
using BSON
using Plots
using Random
using LinearAlgebra
using Flux: mse, ADAM
using Flux: params

# network_type = "RTS_GMLC_risk.m"
#network_type = "edited"
# network_type = "base_case"
# network_type = "sole_gen"
# network_type = "high_risk"
network_type = "hawaii"

## Load the training data
training_data = BSON.load("wildfire_training_data_$network_type.bson")
global keys_of_dictionary = keys(training_data)
global input_data = []
global target_data = []

# Extract input and target data from the dictionary
for i in keys_of_dictionary
    if i != "load_shed"
        push!(input_data, training_data["$i"])
    end
end

# Define a tolerance levela
tolerance = 1e-1

input_data = hcat(input_data...)
target_data = hcat(training_data["load_shed"])

# Round all numbers approximately equal to zero to 0
for i in 1:size(target_data, 1)
    for j in 1:size(target_data, 2)
        if abs(target_data[i, j]) < tolerance  # Check if the element is close to zero
            target_data[i, j] = 0.0
        end
    end
end

function train_test_split(data, test_ratio=0.2, seed=42)
    Random.seed!(seed)  # Set a seed for reproducibility
    indices = randperm(size(data, 1))
    n_test = Int(floor(test_ratio * length(indices)))
    return data[indices[1:end-n_test], :], data[indices[end-n_test+1:end], :]
end

# Split the data into training nd testing datasets
x_train, x_test = train_test_split(input_data, 0.2, 42)
y_train, y_test = train_test_split(target_data, 0.2 , 42)
x_train = x_train'
y_train = y_train'
x_test = x_test'
y_test = y_test'
#sss
# Define the model parameters
input_layer_size = size(x_train)[1]
output_layer_size = size(y_train)[1]
model = Chain(
        Dense(input_layer_size, 100, relu),
        Dense(100, output_layer_size)
)

# Hyperparameters
epochs = 5000
learning_rate = 0.001
patience = 20  # Number of epochs to wait before early stopping if no improvement

# Define the loss function with L2 regularization
function loss_with_regularization(x, y)
    loss_val = Flux.mae(model(x), y)
    λ = 0.1  # Regularization parameter, adjust as needed
    reg_term = sum(norm(p)^2 for p in params(model))  # L2 regularization term
    return loss_val + λ * reg_term
end

# Train the neural network with L2 regularization
# loss(x, y) = loss_with_regularization(x, y)
loss(x, y) = Flux.mae(model(x), y)
opt = ADAM(learning_rate)
data = [(x_train, y_train)]

global best_loss = Inf
global wait = 0
Epoch = []
Loss = []

# Record starting time
start_time = time()

for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), data, opt)
    val_loss = loss(x_test, y_test)
    println("Epoch $epoch, Loss: $val_loss")
    push!(Epoch, epoch)
    push!(Loss, val_loss)

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

# Calculate training time
training_time = time() - start_time
println("Training time: $training_time seconds")

predictions = model(x_test)
errors = predictions - y_test
relative_error = (norm(errors)/norm(y_test)) * 100
println("Relative error of prediction: $relative_error ")

# Function to save the trained model to a BSON file
function save_model_to_bson(filename, model)
    BSON.@save filename model=model
end

# Save the trained model to a BSON file
model_filename = "wildfire_trained_model_$network_type.bson"
save_model_to_bson(model_filename, model)
println("Training and saving completed.")

epoch_combined = vcat(Epoch...)
loss_combined = vcat(Loss...)
plot(epoch_combined, loss_combined, title = "Variation of Validation Loss with Epoch", xlabel = "Epoch number", ylabel = "Mean Absolute Error", label="Relative Error = $relative_error")

