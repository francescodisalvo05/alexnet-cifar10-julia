using MLDatasets
using Flux, Statistics
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using CUDA
using MLDatasets


function get_data(args)
    # it fixes the error "DataType has no batchsize"
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # load train and test dataset
    x_train, y_train = MNIST.traindata()
    x_test,  y_test  = MNIST.testdata()

    # Reshape arbitrarly-shaped input into a matrix-shaped output
    x_train, x_test = Flux.flatten(x_train), Flux.flatten(x_test)

    # One-hot-encode the labels
    y_train, y_test = onehotbatch(y_train, 0:9), onehotbatch(y_test, 0:9)

    train_loader = DataLoader((x_train, y_train), batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader((x_test, y_test), batchsize=args.batchsize)

    return train_loader, test_loader

end


function loss_function(ŷ, y)
    logitcrossentropy(ŷ, y)
end


function evaluation_loss_accuracy(loader, model)

    loss, accuracy, counter = 0f0, 0f0, 0

    for (x,y) in loader
        ŷ = model(x)
        loss += loss_function(ŷ,y)
        accuracy += sum(onecold(ŷ) .== onecold(y))
        counter +=  size(x)[end]
    end

    return loss / counter, accuracy / counter
end


function set_model()
end


# arguments for the `train` function 
Base.@kwdef mutable struct Args
    η = 1e-4             # learning rate
    batchsize = 128      # batch size
    epochs = 10          # number of epochs
end

function train(; kws...)
    args = Args(; kws...) # collect options in a struct for convenience
    
    train_loader, test_loader = get_data(args)

    model = set_model()

    ps = Flux.params(model)  
    opt = ADAM(args.η) 

    for epoch in 1:args.epochs

        for (x,y) in train_loader
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss_function(ŷ, y)
            end

            Flux.Optimise.update!(opt, ps, gs)
        end

        train = evaluation_loss_accuracy(train_loader, model)
        test = evaluation_loss_accuracy(test_loader, model)

        println("Epoch $(epoch-1)")
        println("\t Train => loss = $(train[1]) \t acc = $(train[2])")
        println("\t Test => loss = $(test[1]) \t acc = $(test[2])")
    
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    train()
end