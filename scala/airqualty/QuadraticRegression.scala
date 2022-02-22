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
@main def QuadraticTest10 (): Unit =
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
    val mod   = PolyRegression (airqual,  2, null, PolyRegression.hp)
    mod.trainNtest ()()                                            // train and test the model

    banner ("test for collinearity")
    println ("corr = " + mod.getX.corr)
    println ("vif  = " + mod.vif ())

    banner ("test predictions")
    val yp = x.map (mod.predict (_))
    println (s" y = $y \n yp = $yp")
//    new Plot (x, y, yp, "PolyRegression")
//
//    val z = 10.5                                                   // predict y for one point
//    val yp2 = mod.predict (z)
//    println (s"predict ($z) = $yp2")

    banner ("test cross-validation")
    val stats = mod.crossValidate ()
    Fit.showQofStatTable (stats)

}
end QuadraticTest10