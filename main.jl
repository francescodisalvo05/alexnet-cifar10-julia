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
    x_train, y_train = FashionMNIST.traindata(Float32)
    x_test,  y_test  = FashionMNIST.testdata(Float32)

    # reshape to 1 single channel images
    x_train = reshape(x_train, 28, 28, 1, :)
    x_test = reshape(x_test, 28, 28, 1, :)

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


function set_model(; imgsize=(28,28,1), nclasses=10)
    return Chain(
                Conv((11, 11), imgsize[end]=>64, stride=4, relu),
                MaxPool((3, 3), stride=2),
                Conv((5, 5), 64=>192, stride=4, relu),
                MaxPool((3, 3), stride=2),
                Conv((5, 5), 192=>284, relu, pad=(1,1)),
                MaxPool((3, 3), stride=2),
                Conv((3, 3), 384=>256, relu, pad=(1,1)),
                Conv((3, 3), 256=>256, relu, pad=(1,1)),
                MaxPool((3, 3), stride=2),
                MeanPool((6, 6)),
                flatten,
                Dense(4096, 4096, relu),
                Dropout(0.5),
                Dense(4096, 4096, relu),
                Dropout(0.5),
                Dense(4096, nclasses, softmax))
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
            gs = gradient(() -> loss_function(model(x), y), ps) # compute gradient
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