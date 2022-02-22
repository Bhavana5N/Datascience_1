package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.{x, x_fname, xy, y}

import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf



@main def QuadraticTest9 (): Unit =
{
  val mod   = PolyRegression (xy,  2, null, PolyRegression.hp)
  mod.trainNtest ()()                                            // train and test the model

  banner ("test for collinearity")
  println ("corr = " + mod.getX.corr)
  println ("vif  = " + mod.vif ())

  banner ("test predictions")

  banner ("test cross-validation")
  val stats = mod.crossValidate ()
  Fit.showQofStatTable (stats)
  for tech <- Predictor.SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)                      // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Symbolic Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

}
end QuadraticTest9

