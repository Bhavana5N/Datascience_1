package scalation
package modeling

import scalation.mathstat._



@main def symLassoRegressionTest10 (): Unit =

  import scala.collection.mutable.Set

  //  println (s"x = $x")
  //  println (s"y = $y")
  val housing = MatrixD.load("USA_Housing.csv")

  val n = housing.dim2 - 1
  val (x, y) = (housing.not(?, n), housing(?, n))
  val x_fname= Array ("Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", 
                            "Avg. Area Number of Bedrooms","Area Population")
  val mod = SymLassoRegression (x, y, x_fname, scala.collection.immutable.Set (1.0,1.0,1.0, 1.0), false, true)                // add x^2 terms
  mod.trainNtest ()()                                                   // train and test the model
                                            // parameter/coefficient statistics

  for tech <- Predictor.SelectionTech.values do
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
      s"R^2 vs n for Symbolic Lasso Regression with $tech", lines = true)
    println (s"$tech: rSq = $rSq")
  end for
  println(mod.summary())
  println(mod.report(mod.test ()._2))

end symLassoRegressionTest10