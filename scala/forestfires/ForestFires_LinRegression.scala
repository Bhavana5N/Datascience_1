package scalation
package modeling

import forestfires_data._

@main def ForestFires_SimplerRegression (): Unit =

    banner ("SimplerRegression model: y = b₁*x₁")
    val mod = SimplerRegression (xy)                                   // create a SimplerRegression model
    mod.trainNtest ()()                                                // train and test the model

end ForestFires_SimplerRegression

@main def ForestFires_Regression (): Unit =

    banner ("Regression model: y = b₀ + b₁*x₁ + b₂*x₂ + b₃*x₃ + b₄*x₄ + b₅*x₅ + b₆*x₆")
    val mod = Regression (oxy, ox_fname)()                             // create a Regression Model (with intercept)
    mod.trainNtest ()()                                                // train and test the model
    println (mod.summary ())                                           // produce summary statistics

    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

end ForestFires_Regression

