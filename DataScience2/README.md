# Datascience_1

## DataSets:
  winequality.csv- Both for Python and Scala
  
  auto-mpg.csv  - Only for Python. For Scala Example AutoMpG Data
  
  Expedia.csv - Only for Python, In Scala folder a dataset is created
  
  forestfires.csv - Only for Python, In Scala folder a dataset is created
  
  USA_Housing.csv - Both for Python and Scala
  


## Python Modules:

  For each dataset a jupyter file is created.
  Data has to be taken from Dataset Folder.
  And makesure data folder is present in same folder as Jupyter notebook file
  Execute ipynb file 

## AutoMPG: \python\autompg
   Neural_autompg_2L
    Neural Net 2 Layers
   Neural_autompg_3L
       Neural Net 2 Layers
   Neural_autompg_XL
       Neural Net 2 Layers

  
## Scala Modules:
  For each Dataset we have seperate file for each model
  Dataset files has to be placed in data folder in myscalation folder
  Models has to be placed in modeling folder
  For Symbolic Lasso Regression a new file is created in scala folder which has to be placed in modeling
  Then Execute each model to get ouput
  
Instructions for running the Scala code.

Copy all the scala codes to the below folder
'scalation_2.0\src\main\scala\scalation\modeling\neuralnet'

Copy the Expedia.csv file into the below folder
scalation_2.0\data.

Go to scalation_2.0 in command terminal and follow the below step,
1) sbt
2) compile
3) runMain scalation.modeling."main_class name". 
   For example, runMain scalation.modeling.NeuralNet3L_Expedia 
## Wine Quality Dataset Scala Function Names:
Wine_Neural2L
runMain scalation.modeling.Wine_Neural2L

Wine_Neural3L
runMain scalation.modeling.Wine_Neural3L

WineNeural_XL
runMain scalation.modeling.WineQuality_XL

## USA_Housing Scala Function Names:
Housing_Neural2L
runMain scalation.modeling.Housing_Neural2L

Housing_Neural3L
runMain scalation.modeling.Housing_Neural3L

Housing_NeuralXL
runMain scalation.modeling.Housing_NeuralXL

## AutoMPG Scala Function Names:

autompg_neuralnet2L
runMain scalation.modeling.NeuralNet2L_autom

autompg_neuralnet3L
runMain scalation.modeling.NeuralNet3L_autom

autompg_neuralnetXL
runMain scalation.modeling.NeuralNetXL_autom


## ForestFires Scala Function Names:

runMain scalation.modeling.NeuralNet2L_Forestfires

runMain scalation.modeling.NeuralNet3L_Forestfires

runMain scalation.modeling.NeuralNetXL_Forestfires

## Expedia Scala Function Names:

runMain scalation.modeling.NeuralNet2L_Expedia

runMain scalation.modeling.NeuralNet3L_Expedia


runMain scalation.modeling.NeuralNetXL_Expedia



