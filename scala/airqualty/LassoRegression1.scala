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
@main def lassoRegressionTest1 (): Unit =
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
    val mod = new LassoRegression (x, y)
    mod.trainNtest ()()
    println (mod.summary ())                                       // parameter/coefficient statistics
   //println (s"predict ($z) = ${mod.predict (z)}")                 // make an out-of-sample prediction
    val yyp = mod.predict (x)                                      // predict y for several points
    println (s"predict (x) = $yyp")
    banner ("Forward Selection Test")
    mod.forwardSelAll (cross = true)
    banner ("Backward Elimination Test")
    mod.backwardElimAll (cross = true)

}
end lassoRegressionTest1