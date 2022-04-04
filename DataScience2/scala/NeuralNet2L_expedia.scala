package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.neuralnet.*
import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf

import ActivationFun._


@main def NeuralNet2L_Expedia (): Unit =
{
 
    import Expedia_data._
//  println (s"ox = $ox")
//  println (s"y  = $y")

    println (s"ox_fname = ${stringOf (ox_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    banner ("Expedia NeuralNet_2L")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
//  println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                     // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")


    for f <- f_aff do                                            // try all activation functions for first layer
        banner (s"Expedia NeuralNet_2L with ${f.name}")
        val mod = NeuralNet_2L.rescale (ox, yy, ox_fname, f = f)  // create model with intercept (else pass x) - rescales
        mod.trainNtest2 ()()                                      // train and test the model - with auto-tuning

        banner ("Expedia Validation Test")
        println (Fit.showFitMap (mod.validate ()()))
    end for

}
end NeuralNet2L_Expedia


