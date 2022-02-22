package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.{x, x_fname, y}

import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf


@main def LasoTest8 (): Unit =
{

  val vgsales = MatrixD.load("Foldspp.csv")

  val n = vgsales.dim2 - 1
  val (x, y) = (vgsales.not(?, n), vgsales(?, n))
  println(x)
  println(y)
  println (s"x = $x")
  println (s"y = $y")

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


  banner ("Forward Selection Test")
  mod.forwardSelAll (cross = false)

  banner ("Backward Elimination Test")
  mod.backwardElimAll (cross = false)

}
end LasoTest8