package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.{x, x_fname, y}

import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf



@main def QuadraticTest8 (): Unit =
{

  val vgsales = MatrixD.load("Foldspp.csv")

  val n = vgsales.dim2 - 1
  val (x, y) = (vgsales.not(?, n), vgsales(?, n))
  println(x)
  println(y)
  val mod   = PolyRegression (vgsales,  2, null, PolyRegression.hp)
  mod.trainNtest ()()                                            // train and test the model

  banner ("test for collinearity")
  println ("corr = " + mod.getX.corr)
  println ("vif  = " + mod.vif ())

  banner ("test predictions")
  val yp = x.map (mod.predict (_))
  println (s" y = $y \n yp = $yp")

  banner ("test cross-validation")
  val stats = mod.crossValidate ()
  Fit.showQofStatTable (stats)

}
end QuadraticTest8