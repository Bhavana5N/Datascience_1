package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.neuralnet.*
import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf


import ActivationFun._

@main def NeuralNetXL_Forestfires (): Unit =
{
    import forestfires_data._                       // don't include intercept, uses biases instead

    println (s"x_fname = ${stringOf (x_fname)}")

    val yy = MatrixD (y).transpose                               // vector to matrix with 1 column

    banner ("Forestfires NeuralNet_XL")
//  val mod = new NeuralNet_XL (x, yy, x_fname)                  // create model with intercept (else pass x)
    val mod = NeuralNet_XL.rescale (x, yy, x_fname)              // create model with intercept (else pass x) - rescales
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
//  println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("Forestfires Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                     // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

    Optimizer.hp ("eta") = 0.005                                 // some activation functions need smaller eta
    for f <- f_aff do                               // try all activation functions for first two layers
        banner (s"Forestfires NeuralNet_XL with ${f.name}")
        val mod = NeuralNet_XL.rescale (x, yy, x_fname,
                             f = Array (f, f, f_id))            // create model with intercept (else pass x) - rescales
        mod.trainNtest2 ()()                                     // train and test the model - with auto-tuning

        banner ("Forestfires Validation Test")
        println (Fit.showFitMap (mod.validate ()()))
    end for
    }

end NeuralNetXL_Forestfires