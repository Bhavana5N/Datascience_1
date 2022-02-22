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
@main def RidgeRegressionTest9 (): Unit =
{
    import RidgeRegression.hp
    println (s"hp = $hp")
    val hp2 = hp.updateReturn ("lambda", 1.0)                          // try different values
    println (s"hp2 = $hp2")
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
    var mod = new RidgeRegression (x, y)                               // create a regression model
    mod.trainNtest ()()                                            // train the model
    println (mod.report (mod.test ()._2))

    println (s"x = $x")
    println (s"y = $y")

    // Compute centered (zero mean) versions of x and y
    val mu_x = x.mean                                                  // column-wise mean of x
    val mu_y = y.mean                                                  // mean of y
    val x_c  = x - mu_x                                                // centered x (column-wise)
    val y_c  = y - mu_y

    banner ("Optimize lambda")
    println (s"findLambda2 = ${mod.findLambda2 ()}")

}
end RidgeRegressionTest9