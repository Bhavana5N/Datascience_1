package scalation
package modeling

import scala.math.sqrt
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._
//import scalation.minima.GoldenSectionLS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegression` class supports multiple linear ridge regression.
 *  In this case, x is multi-dimensional [x_1, ... x_k].  Ridge regression puts
 *  a penalty on the L2 norm of the parameters b to reduce the chance of them taking
 *  on large values that may lead to less robust models.  Both the input matrix x
 *  and the response vector y are centered (zero mean).  Fit the parameter vector
 *  b in the regression equation
 *      y  =  b dot x + e  =  b_1 * x_1 + ... b_k * x_k + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve for the parameter vector b
 *  using the regularized Normal Equations:
 *      b  =  fac.solve (.)  with regularization  x.t * x + λ * I
 *  Five factorization techniques are provided:
 *      'QR'         // QR Factorization: slower, more stable (default)
 *      'Cholesky'   // Cholesky Factorization: faster, less stable (reasonable choice)
 *      'SVD'        // Singular Value Decomposition: slowest, most robust
 *      'LU'         // LU Factorization: similar, but better than inverse
 *      'Inverse'    // Inverse/Gaussian Elimination, classical textbook technique 
 *  @see statweb.stanford.edu/~tibs/ElemStatLearn/
 *  @param x       the centered data/input m-by-n matrix NOT augmented with a first column of ones
 *  @param y       the centered response/output m-vector
 *  @param fname_  the feature/variable names
 *  @param hparam  the shrinkage hyper-parameter, lambda (0 => OLS) in the penalty term 'lambda * b dot b'
 */
@main def ridgeRegressionTest50 (): Unit =

    val housing = MatrixD.load("USA_Housing.csv")
    val n = housing.dim2 - 1
    val (x, y) = (housing.not(?, n), housing(?, n))
    var mod = new Regression (x, y)                               // create a regression model
    mod.train ()                                              // train the model
    println (mod.report (mod.test ()._2))
    for tech <- Predictor.SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                    // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for RidgeRegression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
    println(mod.summary())
    println(mod.report(mod.test ()._2))

end ridgeRegressionTest50