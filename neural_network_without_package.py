from activation_functions import sigmoid


def forward_propagation(
    X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
):
    # Hidden layer calculate
    hidden_activations = []
    for i in range(len(weights_input_hidden)):
        z_hidden = (
            sum(w * x for w, x in zip(weights_input_hidden[i], X)) + bias_hidden[i]
        )
        a_hidden = sigmoid(z_hidden)
        hidden_activations.append(a_hidden)

    z_output = (
        sum(w * x for w, x in zip(weights_hidden_output, hidden_activations))
        + bias_output
    )
    a_output = sigmoid(z_output)
    return a_output


def back_propagation(
    X, y, lr, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
):
    hidden_activations = []
    for i in range(len(weights_input_hidden)):
        z_hidden = (
            sum(w * x for w, x in zip(weights_input_hidden[i], X)) + bias_hidden[i]
        )
        a_hidden = sigmoid(z_hidden)
        hidden_activations.append(a_hidden)

    z_output = (
        sum(w * x for w, x in zip(weights_hidden_output, hidden_activations))
        + bias_output[0]
    )
    a_output = sigmoid(z_output)

    error_output = a_output - y
    delta_output = error_output * (1 - a_output) * a_output
    delta_hidden = [
        delta_output * (1 - a) * a * w
        for w, a in zip(weights_hidden_output, hidden_activations)
    ]

    for i in range(len(weights_hidden_output)):
        weights_hidden_output[i] -= lr * delta_output * hidden_activations[i]
    bias_output[0] -= lr * delta_output

    for i in range(len(weights_input_hidden)):
        for j in range(len(weights_input_hidden[i])):
            weights_input_hidden[i][j] -= lr * delta_hidden[i] * X[j]
        bias_hidden[i] -= lr * delta_hidden[i]

    return
