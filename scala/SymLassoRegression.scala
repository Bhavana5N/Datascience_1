package scalation.modeling

import scalation.mathstat.{MatrixD, VectorD}

object SymLassoRegression:

  //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /**
   *  @param x       the initial data/input m-by-n matrix (before expansion)
   *                     must not include an intercept column of all ones
   *  @param y       the response/output m-vector
   *  @param fname   the feature/variable names (use null for default)
   *  @param powers  the set of powers to raise matrix x to
   *  @param cross   whether to include 2-way cross/interaction terms x_i x_j (defaults to true)
   *  @param cross3  whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
   *  @param hparam  the hyper-parameters (use RidgeRegression.hp for default)
   */
  def apply (x: MatrixD, y: VectorD, fname: Array [String],
             powers: Set [Double], cross: Boolean = true, cross3: Boolean = false,
             hparam: HyperParameter = LassoRegression.hp): LassoRegression =
    var xx     = x                                                    // start with multiple regression for x
    var fname_ = fname

    for p <- powers if p != 1 do
      xx       = xx ++^ x~^p                                        // add power terms x^p
      fname_ ++= fname.map ((n) => s"$n^${p.toInt}")
    end for

    if cross then
      xx       = xx ++^ x.crossAll                                   // add 2-way cross terms x_i x_j
      fname_ ++= SymbolicRegression.crossNames (fname)
    end if

    if cross3 then
      xx       = xx ++^ x.crossAll3                                  // add 3-way cross terms x_i x_j x_k
      fname_ ++= SymbolicRegression.crossNames3 (fname)
    end if

    val mod = new LassoRegression (xx, y, fname_, hparam)
    mod.modelName = "SymLassoRegression" + (if cross then "X" else "") + (if cross3 then "XX" else "")
    mod
  end apply



end SymLassoRegression