package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.WineQuality_data.{x, x_fname, xy, y, ox_fname, ox}
import scalation.modeling.neuralnet.*
import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf
import java.io.PrintStream
import java.io.FileOutputStream
import ActivationFun._



@main def Wine_Neural3L (): Unit =
{
 
//  println (s"ox = $ox")
//  println (s"y  = $y")
//    System.setOut(new PrintStream(new FileOutputStream("log/Output/3L_NN_WineQuality.txt")))
    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column
    //System.setOut(new PrintStream(new FileOutputStream("log/Output/nn_3L_WineQuality.txt")))
    banner ("Wine Quality NeuralNet_3L")
    val mod = NeuralNet_3L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

    println (s"ox_fname = ${stringOf (ox_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)              // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${ox.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
    //for f <- f_aff do                                            
        //banner (s"Wine Quality NeuralNet_3L with ${f.name}")
        //val mod = NeuralNet_3L.rescale (ox, yy, ox_fname, f = f)  // create model with intercept (else pass x) - rescales
        //mod.trainNtest2 ()()                                      // train and test the model - with auto-tuning
        
        //banner ("Wine Quality Validation Test")
        //println (Fit.showFitMap (mod.validate ()()))
    //end for


}
end Wine_Neural3L

