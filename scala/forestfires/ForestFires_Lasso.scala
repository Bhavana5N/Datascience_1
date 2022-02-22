package scalation
package modeling

import scala.math.sqrt
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

@main def Forestfires_LassoRegression (): Unit =

    import forestfires_data._

    banner ("LassoRegression for Forest fires")
    val mod = new LassoRegression (x, y)                           // create a Lasso regression model
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics

    println (s"best (lambda, sse) = ${mod.findLambda}")
    
    
    banner ("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 Bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())
    
end Forestfires_LassoRegression