using MLDatasets
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using CUDA
using MLDatasets


function get_data(args)

    # load train and test dataset
    x_train, y_train = MNIST.traindata()
    x_test,  y_test  = MNIST.testdata()

    # Reshape arbitrarly-shaped input into a matrix-shaped output
    x_train, x_test = Flux.flatten(x_train), Flux.flatten(x_test)

    train_loader = DataLoader((x_train, y_train), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((x_test, y_test), batchsize=args.batchsize)

    return train_loader, test_loader

end


function loss_function(ŷ, y)
    logitcrossentropy(ŷ, y)
end


function evaluation_loss_accuracy(loader, model, device)

    loss, accuracy, counter = 0f0, 0f0, 0

    for (x,y) in loader
        ŷ = model(x)
        loss += loss_function(ŷ,y)
        accuracy += sum(onecold(ŷ) .== onecold(y))
        counter +=  size(x)[end]
    end

    return loss / counter, accuracy / counter
end

