package scalation
package modeling

import scala.math.sqrt
import scala.runtime.ScalaRunTime.stringOf

import Expedia_data._
import scalation.mathstat.PlotM


@main def Expedia_ridgeRegression (): Unit =

    import Expedia_data._

    // banner ("forestfires Regression")
    // val reg = new Regression (ox, y, ox_fname)                         // create a regression model (with intercept)
    // reg.trainNtest ()()                                                // train and test the model
    // println (reg.summary ())                                           // parameter/coefficient statistics

//  println (s"x = $x")                                                // data matrix without intercept
//  println (s"y = $y")                                                // response vector

    banner ("Expedia Ridge Regression")
    val mod = RidgeRegression (x, y, x_fname, RidgeRegression.hp)      // create a ridge regression model (no intercept)
    mod.trainNtest ()()                                                // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics


    println (s"x_fname = ${stringOf (x_fname)}")

    for tech <- Predictor.SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                    // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for RidgeRegression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())
    
end Expedia_ridgeRegression



