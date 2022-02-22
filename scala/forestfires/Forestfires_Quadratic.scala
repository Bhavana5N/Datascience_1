package scalation
package modeling

import forestfires_data._

@main def ForestFires_QuadraticRegression (): Unit =
    
    val order = 2
    val mod   = PolyRegression (xy, order, null, PolyRegression.hp)
    mod.trainNtest ()()                                            // train and test the model
    println (mod.summary ()) 


    banner ("Cross-Validation")
    Fit.showQofStatTable (mod.crossValidate ())

end ForestFires_QuadraticRegression