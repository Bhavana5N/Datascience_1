package scalation
package modeling
import scalation.mathstat.MatrixD
import scalation.mathstat._
//import scalation.modeling.Imputation.x2
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.x
import scalation.modeling.ImputeMean
import scala.collection.mutable.ArrayBuffer
//import scala.collection.mutable._



@main def symLassoRegressionTest11 (): Unit =
{
  import scala.collection.mutable.Set

  banner ("Symbolic Regression")
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
    val x_fname= Array ("CO(GT)", "PT08.S1(CO)", "NMHC(GT)", 
                            "C6H6(GT)","PT08.S2(NMHC)","NOx(GT)",
                            "PT08.S3(NOx)","PT08.S4(NO2)","PT08.S5(O3)",
                            "T","RH","AH")
    val mod = SymLassoRegression (x, y, x_fname, scala.collection.immutable.Set (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), false, true)                // add x^2 terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- Predictor.SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
        s"R^2 vs n for  Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
    println(mod.report (mod.test ()._2))
}
end symLassoRegressionTest11