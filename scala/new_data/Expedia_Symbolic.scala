package scalation
package modeling

import scalation.mathstat.PlotM

import scala.collection.mutable.Set
import scala.runtime.ScalaRunTime.stringOf

import Expedia_data._

@main def Expedia_symbolicRegression (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("Expedia Symbolic Regression")
    val mod = SymbolicRegression.quadratic (x, y, x_fname)   // add, intercept, cross-terms and given powers
    mod.trainNtest ()()                                                  // train and test the model
    println (mod.summary ())                                             // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                      // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Symbolic Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())
    
end Expedia_symbolicRegression



