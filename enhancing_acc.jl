# It sounds like you're dealing with a regression problem where the output variable (load shed) ranges between 0 and 2. Since you mentioned that treating it as a regression problem isn't giving satisfactory results, and considering the nature of your data, you might want to try a different approach.

# One approach you could try is transforming your regression problem into a classification one by discretizing the output variable. You can define thresholds to classify the load shed into different categories, for example, 0-0.5 as class 0, 0.5-1.5 as class 1, and 1.5-2 as class 2. Then, you can train your neural network to predict these classes instead of the continuous load shed value.

using Flux
using Flux: onehotbatch, crossentropy, throttle
using Base.Iterators: repeated
using Statistics

# Assuming `X` is your input data and `Y` is your output data (load shed)
# Normalize your input data if necessary
# X_normalized = normalize(X)

# Define your thresholds for classification
thresholds = [0.5, 1.5]

# Function to convert load shed values to class labels
function shed_to_class(shed_value)
    if shed_value <= thresholds[1]
        return 1
    elseif thresholds[1] < shed_value <= thresholds[2]
        return 2
    else
        return 3
    end
end

# Convert load shed values to class labels
Y_classes = shed_to_class.(Y)

# Convert class labels to one-hot encoding
Y_onehot = onehotbatch(Y_classes, 1:3)

# Define your neural network model
model = Chain(
    Dense(input_size, hidden_size, relu),
    Dense(hidden_size, output_size),
    softmax
)

# Define your loss function (cross-entropy for classification)
loss(x, y) = crossentropy(model(x), y)

# Define your optimizer
opt = ADAM()

# Train your model
data = [(X_normalized[i], Y_onehot[i]) for i in 1:size(X_normalized)[1]]
losses = []
accuracy = []

for epoch in 1:num_epochs
    Flux.train!(loss, params(model), data, opt)
    push!(losses, loss(X_normalized, Y_onehot))
    accuracy_epoch = accuracy_score(argmax(model.(X_normalized), dims=1), Y_classes)
    push!(accuracy, accuracy_epoch)
    println("Epoch $epoch - Loss: $(losses[end]), Accuracy: $accuracy_epoch")
end
