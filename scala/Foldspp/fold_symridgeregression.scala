package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP

import scala.collection.mutable.{ArrayBuffer}
import scala.runtime.ScalaRunTime.stringOf


@main def symRidgeRegressionTest10 (): Unit =

  import scala.collection.mutable.Set

  //  println (s"x = $x")
  //  println (s"y = $y")
  val x_name = Array("AT","V","AP",	"RH")
  val vgsales = MatrixD.load("Foldspp.csv")

  val n = vgsales.dim2 - 1
  val (x, y) = (vgsales.not(?, n), vgsales(?, n))
  banner ("auto_mpg Quadratic Ridge Regression")
  val mod = SymRidgeRegression (x, y, x_name, scala.collection.immutable.Set (1.0,1.0,1.0, 1.0), false, true)                // add x^2 terms
  mod.trainNtest ()()                                                   // train and test the model
  println (mod.summary ())                                              // parameter/coefficient statistics

  for tech <- Predictor.SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Quadratic Ridge Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

end symRidgeRegressionTest10