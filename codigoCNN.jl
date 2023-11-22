
using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using FileIO
using Statistics: mean
using WAV
using Statistics
using FFTW
using Glob
using Plots
senales = [] #matriz de senales
etiquetas = [] #matriz de etiquetas
labels = ["zombie", "no_zombie"]
using Random
using Random: seed!
function normalizeMinMaxAux(value::Real, max::Real, min::Real)
    if (max == min)
        return 0
    else
        return (value - min) / (max - min)
    end
end;
function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N / k)))
    indices = indices[1:N]
    shuffle!(indices)
    return indices
end
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    if (numClasses == 2)
        return reshape((feature .== classes[1]), :, 1)
    else
        encodedTargets = Array{Bool,2}(undef, size(feature, 1), numClasses)
        for i = 1:numClasses
            encodedTargets[:, i] .= (feature .== classes[i])
        end
    end
    return encodedTargets
end;
# Función para generar un array con las señales de frecuencia de un archivo de audio
function generar_array_audios(audio_file)
    audio, Fs = wavread(audio_file)
    senalFrecuencia = abs.(fft(audio))
    return senalFrecuencia
end
seed!(1);
# Ahora array_senales es un array de arrays, donde cada subarray contiene la señal de frecuencia de un archivo de audio
zombie = ["zombie", "zombiepig", "zombie_villlager", "drowned"] #ponemos los que son zombies
nozombie = ["oveja", "spider", "vaca", "villager", "skeleton", "wolf"] #ponemos los que no son zombies
# Carpeta de entrada
#input_folder = "u"
input_folder = "./fonts/outputs/"
# Inicializar un array vacío para almacenar las señales de frecuencia de cada archivo de audio


i = 0
# Iterar sobre las carpetas de la carpeta de entrada
for folder in readdir(input_folder)
    # Obtener lista de archivos OGG en la carpeta actual
    audio_files = glob("*.wav", joinpath(input_folder, folder))
    # Iterar sobre los archivos de audio y almacenar sus señales de frecuencia en el array
    for audio_file in audio_files
        global i
        i += 1
        push!(senales, generar_array_audios(audio_file))
        if folder in zombie
            push!(etiquetas, "zombie")
        else
            push!(etiquetas, "no_zombie")
        end
    end
end
targets = etiquetas
inputs = senales

crossValidationIndices = crossvalidation(size(targets, 1), 10);
print(crossValidationIndices)
# Dividimos los datos en entrenamiento y test
trainingInputs = inputs[crossValidationIndices.!=1]
testInputs = inputs[crossValidationIndices.==1]
train_labels = targets[crossValidationIndices.!=1]
test_labels = targets[crossValidationIndices.==1]
train_audios = hcat(trainingInputs, train_labels)
test_audios = hcat(testInputs, test_labels)

println("Tamaño de la matriz de entrenamiento: ", size(train_audios))
println("Tamaño de la matriz de test:          ", size(test_audios))

# Tanto train_imgs como test_imgs son arrays de arrays bidimensionales (arrays de imagenes), es decir, son del tipo Array{Array{Float32,2},1}
#  Generalmente en Deep Learning los datos estan en tipo Float32 y no Float64, es decir, tienen menos precision
#  Esto se hace, entre otras cosas, porque las tarjetas gráficas (excepto las más recientes) suelen operar con este tipo de dato
#  Si se usa Float64 en lugar de Float32, el sistema irá mucho más lento porque tiene que hacer conversiones de Float64 a Float32

# Para procesar las imagenes con Deep Learning, hay que pasarlas una matriz en formato HWCN
#  Es decir, Height x 1 x 1 x N
#  En el caso de esta base de datos
#   Height = 28
#   Width = 28
#   Channels = 1 -> son imagenes en escala de grises
#     Si fuesen en color, Channels = 3 (rojo, verde, azul)
# Esta conversion se puede hacer con la siguiente funcion:
function convertirArrayImagenesHWCN(senales)
    numPatrones = length(senales)
    numVen = length(senales[1])
    nuevoArray = Array{Float32,4}(undef, numVen, 1, 1, numPatrones) # Importante que sea un array de Float32
    for i in 1:numPatrones
        nuevoArray[:, 1, 1, i] .= senales[i]
    end
    return nuevoArray
end;
train_imgs = convertirArrayImagenesHWCN(trainingInputs);
test_imgs = convertirArrayImagenesHWCN(testInputs);

println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
println("Tamaño de la matriz de test:          ", size(test_imgs))


# Cuidado: en esta base de datos las imagenes ya estan con valores entre 0 y 1
# En otro caso, habria que normalizarlas
println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");
print(size(train_imgs)[1])
min_train = minimum(train_imgs)
max_train = maximum(train_imgs)
for i in 1:size(train_imgs)[4]
    for j in 1:size(train_imgs)[1]
        train_imgs[j, 1, 1, i] = normalizeMinMaxAux(train_imgs[j, 1, 1, i], max_train, min_train)
    end
end
min_test = minimum(test_imgs)
max_test = maximum(test_imgs)
for i in 1:size(test_imgs)[4]
    for j in 1:size(test_imgs)[1]
        test_imgs[j, 1, 1, i] = normalizeMinMaxAux(test_imgs[j, 1, 1, i], max_test, min_test)
    end
end



# Cuando se tienen tantos patrones de entrenamiento (en este caso 60000),
#  generalmente no se entrena pasando todos los patrones y modificando el error
#  En su lugar, el conjunto de entrenamiento se divide en subconjuntos (batches)
#  y se van aplicando uno a uno

# Hacemos los indices para las particiones
# Cuantos patrones va a tener cada particion
batch_size = 185
# Creamos los indices: partimos el vector 1:N en grupos de batch_size
gruposIndicesBatch = Iterators.partition(1:size(train_imgs, 4), batch_size);
println("He creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");


# Creamos el conjunto de entrenamiento: va a ser un vector de tuplas. Cada tupla va a tener
#  Como primer elemento, las imagenes de ese batch
#     train_imgs[:,:,:,indicesBatch]
#  Como segundo elemento, las salidas deseadas (en booleano, codificadas con one-hot-encoding) de esas imagenes
#     Para conseguir estas salidas deseadas, se hace una llamada a la funcion onehotbatch, que realiza un one-hot-encoding de las etiquetas que se le pasen como parametros
#     onehotbatch(train_labels[indicesBatch], labels)
#  Por tanto, cada batch será un par dado por
#     (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels))
# Sólo resta iterar por cada batch para construir el vector de batches
train_set = [(train_imgs[:, :, :, indicesBatch], onehotbatch(train_labels[indicesBatch], labels)) for indicesBatch in gruposIndicesBatch];

# Creamos un batch similar, pero con todas las imagenes de test
test_set = (test_imgs, onehotbatch(test_labels, labels));


# Hago esto simplemente para liberar memoria, las variables train_imgs y test_imgs ocupan mucho y ya no las vamos a usar
train_imgs = nothing;
test_imgs = nothing;
GC.gc(); # Pasar el recolector de basura




funcionTransferenciaCapasConvolucionales = relu;

# Definimos la red con la funcion Chain, que concatena distintas capas
ann = Chain(
    Conv((3, 1), 1 => 12, pad=(1, 1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2, 1)),
    Conv((3, 1), 12 => 24, pad=(1, 1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2, 1)),
    Conv((3, 1), 24 => 24, pad=(1, 1), funcionTransferenciaCapasConvolucionales),
    MaxPool((2, 1)),
    x -> reshape(x, :, size(x, 4)),
    Dense(688128, 2, σ)
)




# Vamos a probar la RNA capa por capa y poner algunos datos de cada capa
# Usaremos como entrada varios patrones de un batch
numBatchCoger = 1;
numImagenEnEseBatch = [12, 6];
# Para coger esos patrones de ese batch:
#  train_set es un array de tuplas (una tupla por batch), donde, en cada tupla, el primer elemento son las entradas y el segundo las salidas deseadas
#  Por tanto:
#   train_set[numBatchCoger] -> La tupla del batch seleccionado
#   train_set[numBatchCoger][1] -> El primer elemento de esa tupla, es decir, las entradas de ese batch
#   train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch] -> Los patrones seleccionados de las entradas de ese batch
entradaCapa = train_set[numBatchCoger][1][:, :, :, numImagenEnEseBatch];
numCapas = length(Flux.params(ann));
println("La RNA tiene ", numCapas, " capas:");
for numCapa in 1:numCapas
    println("   Capa ", numCapa, ": ", ann[numCapa])
    # Le pasamos la entrada a esta capa
    global entradaCapa # Esta linea es necesaria porque la variable entradaCapa es global y se modifica en este bucle
    capa = ann[numCapa]
    salidaCapa = capa(entradaCapa)
    println("      La salida de esta capa tiene dimension ", size(salidaCapa))
    entradaCapa = salidaCapa
end

# Sin embargo, para aplicar un patron no hace falta hacer todo eso.
#  Se puede aplicar patrones a la RNA simplemente haciendo, por ejemplo
ann(train_set[numBatchCoger][1][:, :, :, numImagenEnEseBatch]);




# Definimos la funcion de loss de forma similar a las prácticas de la asignatura
loss(x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y);
# Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));
# Un batch es una tupla (entradas, salidasDeseadas), asi que batch[1] son las entradas, y batch[2] son las salidas deseadas


# Mostramos la precision antes de comenzar el entrenamiento:
#  train_set es un array de batches
#  accuracy recibe como parametro un batch
#  accuracy.(train_set) hace un broadcast de la funcion accuracy a todos los elementos del array train_set
#   y devuelve un array con los resultados
#  Por tanto, mean(accuracy.(train_set)) calcula la precision promedia
#   (no es totalmente preciso, porque el ultimo batch tiene menos elementos, pero es una diferencia baja)
println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100 * mean(accuracy.(train_set)), " %");


# Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
opt = ADAM(0.001);


println("Comenzando entrenamiento...")
mejorPrecision = -Inf;
criterioFin = false;
numCiclo = 0;
numCicloUltimaMejora = 0;
mejorModelo = nothing;

while (!criterioFin)

    # Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
    global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin

    # Se entrena un ciclo
    Flux.train!(loss, Flux.params(ann), train_set, opt)

    numCiclo += 1

    # Se calcula la precision en el conjunto de entrenamiento:
    precisionEntrenamiento = mean(accuracy.(train_set))
    println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100 * precisionEntrenamiento, " %")

    # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
    if (precisionEntrenamiento >= mejorPrecision)
        mejorPrecision = precisionEntrenamiento
        precisionTest = accuracy(test_set)
        println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100 * precisionTest, " %")
        mejorModelo = deepcopy(ann)
        numCicloUltimaMejora = numCiclo
    end

    # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
    if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
        opt.eta /= 10.0
        println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta)
        numCicloUltimaMejora = numCiclo
    end

    # Criterios de parada:

    # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
    if (precisionEntrenamiento >= 0.999)
        println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
        criterioFin = true
    end

    # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
    if (numCiclo - numCicloUltimaMejora >= 10)
        println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
        criterioFin = true
    end
end
