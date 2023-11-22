Los archivos del tipo aproxX.jl se corresponden con el código que se debe ejecutar para obtener los resultados de la aproximación X.
El archivo source.jl es el código con las funciones comunes para los archivos de las distintas aproximaciones.
El archivo codigoCNN.jl permite obtener los resultados de la aproximación de Deep Learning
En la carpeta /datasets se encuentran las bases de datos utilizadas para las distintas aproximaciones (BDaproxX.data)
En la carpeta /fonts se encuentra el código adicional que hemos necesitado durante el desarrollo de la práctica:
  - convertir_frecuencia.py: Lo necesitamos para transformar la frecuencia (a 44100 Hz) de algunos audios que no tenían la misma que la mayoría. Debe especificarse la carpeta con los audios originales para un mob en concreto y la carpeta de salida.
  - fragmentar.py: Permite fragmentar los audios originales. Debe indicarse la carpeta donde están los archivos .ogg de entrada que se desean fragmentar y la carpeta donde se guardarán los archivos fragmentados.
  - media_desviacion.jl: Este es el código empleado para la extracción de características. Se debe especificar la ruta en la que se encuentran los fragmentos (para un mob concreto) para los que queremos obtener media y desviación típica. Descomentando líneas se pueden mostrar las gráficas de frecuencia que se incluyen en la memoria.
  - /outputs: carpeta que incluye los fragmentos de audio de los distintos mobs.
  - /sonidos_minecraft: carpeta que incluye los audios originales de los distintos mobs.