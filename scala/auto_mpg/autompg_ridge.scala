package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.{x, x_fname, y}

import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf



@main def RidgeRegressionTest9 (): Unit =
{
  import RidgeRegression.hp

  println (s"hp = $hp")
  val hp2 = hp.updateReturn ("lambda", 1.0)                          // try different values
  println (s"hp2 = $hp2")

  var mod = new RidgeRegression (x, y)                               // create a regression model
  mod.trainNtest()()                                           // train the model
  println (mod.report (mod.test ()._2))

  println (s"x = $x")
  println (s"y = $y")

  println (mod.summary ())                                           // parameter/coefficient statistics

  banner ("Cross-Validation")
  //Fit.showQofStatTable (mod.crossValidate ())

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
}
end RidgeRegressionTest9