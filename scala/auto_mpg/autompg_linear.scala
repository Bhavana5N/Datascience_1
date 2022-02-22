package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.{ox, ox_fname, x, x_fname, y}

import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf


@main def regressionTest9 (): Unit =
{

  val mod = new Regression (ox, y, ox_fname)                // create model with intercept (else pass x)
  mod.trainNtest ()()                                                // train and test the model
  println (mod.summary ())                                  // parameter/coefficient statistics

  banner ("Cross-Validation")
  Fit.showQofStatTable (mod.crossValidate ())

  println (s"ox_fname = ${stringOf (ox_fname)}")

  for tech <- Predictor.SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)             // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

}
end regressionTest9