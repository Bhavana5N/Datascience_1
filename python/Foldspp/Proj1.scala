package scalation
package modeling

import scalation.mathstat.MatrixD
import scalation.mathstat.*
import scalation.mathstat.MatrixD.DEF_SEP
import scalation.modeling.Example_AutoMPG.{x, x_fname}

import scala.collection.mutable.{ArrayBuffer, Set}
import scala.runtime.ScalaRunTime.stringOf

@main def regressionTest8 (): Unit =
{

    val vgsales = MatrixD.load("Foldspp.csv")

    val n = vgsales.dim2 - 1
    val (x, y) = (vgsales.not(?, n), vgsales(?, n))
    println(x)
    println(y)
    var mod = new Regression (x, y)                               // create a regression model
    mod.trainNtest ()()                                              // train the model
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
end regressionTest8


@main def RidgeRegressionTest8 (): Unit =
{
    import RidgeRegression.hp

    println (s"hp = $hp")
    val hp2 = hp.updateReturn ("lambda", 1.0)                          // try different values
    println (s"hp2 = $hp2")
    val vgsales = MatrixD.load("Foldspp.csv")
    println(vgsales.dim)
    println(vgsales.dim2)
    val n = vgsales.dim2 - 1
    val (x, y) = (vgsales.not(?, n), vgsales(?, n))
    println(x)
    println(y)
    var mod = new RidgeRegression (x, y)                               // create a regression model
    mod.trainNtest()()                                           // train the model
    println (mod.report (mod.test ()._2))

    println (s"x = $x")
    println (s"y = $y")

    println (mod.summary ())                                           // parameter/coefficient statistics

    banner ("Cross-Validation")
    //Fit.showQofStatTable (mod.crossValidate ())

    println (s"x_fname = ${stringOf (x_fname)}")

    for tech <- Predictor.SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                    // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
            s"R^2 vs n for RidgeRegression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
}
end RidgeRegressionTest8

@main def QuadraticTest8 (): Unit =
{

    val vgsales = MatrixD.load("Foldspp.csv")

    val n = vgsales.dim2 - 1
    val (x, y) = (vgsales.not(?, n), vgsales(?, n))
    println(x)
    println(y)
    val mod   = PolyRegression (vgsales,  2, null, PolyRegression.hp)
    mod.trainNtest ()()                                            // train and test the model

    banner ("test for collinearity")
    println ("corr = " + mod.getX.corr)
    println ("vif  = " + mod.vif ())

    banner ("test predictions")
    val yp = x.map (mod.predict (_))
    println (s" y = $y \n yp = $yp")

    banner ("test cross-validation")
    val stats = mod.crossValidate ()
    Fit.showQofStatTable (stats)

}
end QuadraticTest8

@main def LasoTest8 (): Unit =
{

    val vgsales = MatrixD.load("Foldspp.csv")

    val n = vgsales.dim2 - 1
    val (x, y) = (vgsales.not(?, n), vgsales(?, n))
    println(x)
    println(y)
    println (s"x = $x")
    println (s"y = $y")

    banner ("LassoRegression")
    val mod = new LassoRegression (x, y)                           // create a Lasso regression model
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics

    banner ("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 Bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
        "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")


    banner ("Forward Selection Test")
    mod.forwardSelAll (cross = false)

    banner ("Backward Elimination Test")
    mod.backwardElimAll (cross = false)

}
end LasoTest8



@main def symbolicRegressionTest10 (): Unit =

    //  println (s"x = $x")
    //  println (s"y = $y")
    val x_name = Array("AT","V","AP",	"RH")
    val vgsales = MatrixD.load("Foldspp.csv")

    val n = vgsales.dim2 - 1
    val (x, y) = (vgsales.not(?, n), vgsales(?, n))
    val mod = SymbolicRegression (x, y, x_name, Set (1,1,1, 1), false, true)   // add, intercept, cross-terms and given powers
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

end symbolicRegressionTest10