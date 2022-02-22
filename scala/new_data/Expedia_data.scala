package scalation
package modeling

import scalation.mathstat._
import scala.runtime.ScalaRunTime.stringOf

object Expedia_data:
    val xr_fname = Array ("region","accommodation_type","yearly_availability","minimum_nights","number_of_reviews","reviews_per_month","owned_hotels")

    val xyr=MatrixD.load("Expedia.csv",1)

    val oxr = xyr.not(?, 6)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname
    
    val xy = xyr

    private val n = xy.dim2-1                                       // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)

    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x
    
    val x_fname=xr_fname
    val ox_fname: Array [String] = Array ("intercept") ++ xr_fname
end Expedia_data


import Expedia_data._
    
@main def Expedia_Correlation (): Unit =

    banner ("Variable Names in ForestFires Dataset")
    println (s"xr_fname = ${stringOf (xr_fname)}")                     // raw dataset
    println (s"x_fname  = ${stringOf (x_fname)}")                      // origin column removed
    println (s"ox_fname = ${stringOf (ox_fname)}")                     // intercept (1's) added

    banner ("Correlation Analysis: reponse y vs. column x(?, j)")
    for j <- x.indices2 do
        val x_j = x(?, j)
        val correlation = y corr x_j
        val corr2       = correlation * correlation
        println (s"correlation of y vs. x(?, $j) = $correlation \t $corr2 \t ${x_fname(j)}")
        new Plot (x_j, y, null, s"y vs, x(?, $j)")
    end for

end Expedia_Correlation
