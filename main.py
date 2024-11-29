from neural_network_without_package import back_propagation, forward_propagation


def main():
    # Training data
    X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_train = [0, 1, 1, 0]
    lr = 0.2

    # Input to Hidden Layer
    weights_input_hidden = [[0.5, -0.6], [-0.3, 0.8]]
    bias_hidden = [0.1, -0.2]

    # Input to Output layer
    weights_hidden_output = [0.7, -0.5]
    bias_output = [0.5]

    for epoch in range(10000):
        for X, y in zip(X_train, y_train):
            back_propagation(
                X,
                y,
                lr,
                weights_input_hidden,
                bias_hidden,
                weights_hidden_output,
                bias_output,
            )

    for X, y in zip(X_train, y_train):
        print(
            f"Input: {X}, Predicted: {forward_propagation(X, weights_input_hidden, bias_hidden, weights_hidden_output,bias_output[0])}, Actual: {y}"
        )


if __name__ == "__main__":
    main()
