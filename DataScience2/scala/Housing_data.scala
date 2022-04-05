package scalation
package modeling
import scalation.mathstat._
object Housing_data:

    /** the names of the predictor variables; the name of response variable is quality
     */
    val xr_fname = Array ("Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", 
                          "Avg. Area Number of Bedrooms", "Area Population")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("USA_Housing.csv")

    val oxr = xyr.not(?, 5)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (5)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end Housing_data
