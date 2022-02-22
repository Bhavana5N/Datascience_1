package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.{x, x_fname, y}

import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf


@main def LasoTest9 (): Unit =
{



  banner ("LassoRegression")
  val mod = new LassoRegression (x, y)                           // create a Lasso regression model
  mod.trainNtest ()()                                            // train and test the model
  println (mod.summary ())                                       // parameter/coefficient statistics

  banner ("Forward Selection Test")
  val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 Bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, k)                                   // instance index
  new PlotM (t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for LassoRegression", lines = true)
  println (s"rSq = $rSq")


  for tech <- Predictor.SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    //println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for  Lasso Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for

  println (mod.summary ())

}
end LasoTest9

