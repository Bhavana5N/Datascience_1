package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.{x, x_fname, y}

import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf



@main def symbolicRegressionTest11 (): Unit =

  //  println (s"x = $x")
  //  println (s"y = $y")
  val x_name = Array("AT","V","AP",	"RH")
  val vgsales = MatrixD.load("Foldspp.csv")

  val n = vgsales.dim2 - 1
  val (x, y) = (vgsales.not(?, n), vgsales(?, n))
  val mod = SymbolicRegression (x, y, x_name, Set (1,1,1, 1), false, true)   // add, intercept, cross-terms and given powers
  mod.trainNtest ()()                                                  // train and test the model
                                           // parameter/coefficient statistics

  for tech <- Predictor.SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)                      // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Symbolic Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for
  println (mod.summary ())

end symbolicRegressionTest11