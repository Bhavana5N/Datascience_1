package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
//import scalation.modeling.Imputation.x2
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.x
import scalation.modeling.ImputeMean

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Set
@main def RegressionTest12 (): Unit =
{
    val airqual = MatrixD.load("AirQualityUCI.csv",1)
    println(airqual.dim)
    println(airqual.dim2)
    val n = airqual.dim2 - 1
    val (x, y) = (airqual.not(?, n), airqual(?, n))
    for (i <- (0 until airqual.dim2) )
        {
        val z = airqual(?, i)
        val x2= x.copy
        val x3 = x.copy
        //var iv = ( -200.000,-200.000)
        val iv = ImputeMean.impute (z)
        println(iv._2)
        println(z.length)

        for (j <- (0 until z.length) ){
            if (airqual(j, i) == -200.000){
                airqual(j, i) = iv._2
            }
        }
            //println (s"x  = $z")
            //println (airqual(?, i))
            //break()
            //new Plot (airqual(?, i), y)

    }
    airqual.write("test.csv")
    var mod = new Regression (x, y)                               // create a regression model
    mod.train ()                                              // train the model
    println (mod.report (mod.test ()._2))
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
end RegressionTest12