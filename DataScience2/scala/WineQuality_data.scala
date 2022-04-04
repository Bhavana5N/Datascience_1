package scalation
package modeling

import scalation.mathstat._

object WineQuality_data:

    /** the names of the predictor variables; the name of response variable is quality
     */
    val xr_fname = Array ("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
                          "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")

    /** the raw combined data matrix 'xyr'
     */
    val xyr = MatrixD.load("winequality.csv")

    val oxr = xyr.not(?, 11)
    val oxr_fname: Array [String] = Array ("intercept") ++ xr_fname

    val xy = xyr                                                       // use all columns - may cause multi-collinearity

    private val n = xy.dim2 - 1                                        // last column in xy

    val (x, y) = (xy.not(?, n), xy(?, n))                              // (data/input matrix, response column)
    val _1     = VectorD.one (xy.dim)                                  // vector of all ones
    val oxy    = _1 +^: xy                                             // prepend a column of all ones to xy
    val ox     = _1 +^: x                                              // prepend a column of all ones to x

    val x_fname: Array [String] = xr_fname.take (11)
    val ox_fname: Array [String] = Array ("intercept") ++ x_fname

end WineQuality_data
