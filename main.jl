using MLDatasets
using PyPlot

function get_data()

    # load train and test dataset
    x_train, y_train = MNIST.traindata()
    x_test,  y_test  = MNIST.testdata()

    # Reshape arbitrarly-shaped input into a matrix-shaped output
    x_train, x_test = Flux.flatten(x_train), Flux.flatten(x_test)

    train_loader = DataLoader((x_train, y_train), batchsize=128, shuffle=true)
    test_loader = DataLoader((x_test, y_test), batchsize=128)

    return train_loader, test_loader

end

