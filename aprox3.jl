include("source.jl")

# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);
numFolds = 10;
# Cargamos el dataset
dataset = readdlm("./datasets/BDaprox3.data", ',');

#RRNNAA-------------------------------------------------------------------------------------------------------------------------------

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [5];
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. 
#Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el 
#conjunto de validacion, se para el entrenamiento
numRepetitionsAANTraining = 50; # Numero de veces que se va a entrenar la RNA 
#para cada fold por el hecho de ser no determinístico el entrenamiento
# Preparamos las entradas y las salidas deseadas
inputs = convert(Array{Float64,2}, dataset[:, 1:4]);
targets = dataset[:, 5];
crossValidationIndices = crossvalidation(size(inputs, 1), numFolds);
# Normalizamos las entradas, a pesar de que algunas se vayan a utilizar para 
#test
normalizeZeroMean!(inputs);
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("topology: ", topology)
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology = [10];
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("topology: ", topology)
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology = [15];
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("topology: ", topology)
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology = [20];
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("topology: ", topology)
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology = [5,15];
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("topology: ", topology)
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology = [10,15];
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("topology: ", topology)
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology = [15,15];
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("topology: ", topology)
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

topology = [15,20];
# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;

modelHyperparameters["numExecutions"] = numRepetitionsAANTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
println("topology: ", topology)
modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

#SVM-------------------------------------------------------------------------------------------------------------------------------
# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 0.01;
C = 100

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
println("C: ", C)
println(", kernel: ", kernel)
println(", kernelDegree: ", kernelDegree)
println(", kernelGamma: ", kernelGamma)
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 0.01;
C = 1000

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
println("C: ", C)
println(", kernel: ", kernel)
println(", kernelDegree: ", kernelDegree)
println(", kernelGamma: ", kernelGamma)
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 0.01;
C = 10000

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
println("C: ", C)
println(", kernel: ", kernel)
println(", kernelDegree: ", kernelDegree)
println(", kernelGamma: ", kernelGamma)
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 0.001;
C = 10000

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
println("C: ", C)
println(", kernel: ", kernel)
println(", kernelDegree: ", kernelDegree)
println(", kernelGamma: ", kernelGamma)
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 0.0001;
C = 10000

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
println("C: ", C)
println(", kernel: ", kernel)
println(", kernelDegree: ", kernelDegree)
println(", kernelGamma: ", kernelGamma)
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 0.0001;
C = 1000

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
println("C: ", C)
println(", kernel: ", kernel)
println(", kernelDegree: ", kernelDegree)
println(", kernelGamma: ", kernelGamma)
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 0.0001;
C = 100000

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
println("C: ", C)
println(", kernel: ", kernel)
println(", kernelDegree: ", kernelDegree)
println(", kernelGamma: ", kernelGamma)
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

# Parametros del SVM
kernel = "rbf";
kernelDegree = 2;
kernelGamma = 0.001;
C = 100000

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
println("C: ", C)
println(", kernel: ", kernel)
println(", kernelDegree: ", kernelDegree)
println(", kernelGamma: ", kernelGamma)
modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

#ARBOLES DE DECISION---------------------------------------------------------------------------------------------------------------

# Parametros de arboles de decision
maxDepth = 1;

# Entrenamos los arboles de decision
println("maxDepth: ", maxDepth)
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);

# Parametros de arboles de decision
maxDepth = 2;

# Entrenamos los arboles de decision
println("maxDepth: ", maxDepth)
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);

# Parametros de arboles de decision
maxDepth = 3;

# Entrenamos los arboles de decision
println("maxDepth: ", maxDepth)
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);

# Parametros de arboles de decision
maxDepth = 4;

# Entrenamos los arboles de decision
println("maxDepth: ", maxDepth)
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);

# Parametros de arboles de decision
maxDepth = 5;

# Entrenamos los arboles de decision
println("maxDepth: ", maxDepth)
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);

# Parametros de arboles de decision
maxDepth = 6;

# Entrenamos los arboles de decision
println("maxDepth: ", maxDepth)
modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);

#KNN-------------------------------------------------------------------------------------------------------------------------------

# Parapetros de KNN
numNeighbors = 1;

# Entrenamos los kNN
println("numNeighbors: ", numNeighbors)
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);

# Parapetros de KNN
numNeighbors = 2;

# Entrenamos los kNN
println("numNeighbors: ", numNeighbors)
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);

# Parapetros de KNN
numNeighbors = 3;

# Entrenamos los kNN
println("numNeighbors: ", numNeighbors)
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);

# Parapetros de KNN
numNeighbors = 4;

# Entrenamos los kNN
println("numNeighbors: ", numNeighbors)
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);

# Parapetros de KNN
numNeighbors = 5;

# Entrenamos los kNN
println("numNeighbors: ", numNeighbors)
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);

# Parapetros de KNN
numNeighbors = 6;

# Entrenamos los kNN
println("numNeighbors: ", numNeighbors)
modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);