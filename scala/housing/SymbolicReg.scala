package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.x

import scala.collection.mutable.ArrayBuffer
import scalation.optimization.LassoAdmm
import scala.collection.mutable.Set

@main def symbolicRegressionTest100 (): Unit =
{
    banner ("Symbolic Regression")
    val housing = MatrixD.load("USA_Housing.csv")
    val n = housing.dim2 - 1
    val (x, y) = (housing.not(?, n), housing(?, n))
    val x_fname= Array ("Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", 
                            "Avg. Area Number of Bedrooms","Area Population")

    println(x)
    println(y)
    val mod = SymbolicRegression (x, y, x_fname, Set(1, 1, 1, 1, 1), false, true)          // add intercept, cross-terms and given powers
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
    println(mod.summary())
    println(mod.report(mod.test ()._2))
}
end symbolicRegressionTest100

