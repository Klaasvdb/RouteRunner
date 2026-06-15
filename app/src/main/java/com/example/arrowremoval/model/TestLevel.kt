package com.example.arrowremoval.model

/**
 * Test level defined as a string grid:
 *   '#' = wall
 *   '>' = right arrow  (checks rightward for blocking arrows)
 *   '<' = left arrow   (checks leftward for blocking arrows)
 *   '^' = up arrow     (checks upward for blocking arrows)
 *   'v' = down arrow   (checks downward for blocking arrows)
 *   ' ' = empty corridor
 *
 * Solvable removal order (two waves):
 *   Wave 1 — immediately removable (clear line to grid edge):
 *     < at (2,4), < at (8,4)     → no arrows to their left
 *     > at (2,16), > at (8,16)   → no arrows to their right
 *     ^ at (4,2), ^ at (6,2)     → no arrows above them in col 2
 *     v at (4,18), v at (6,18)   → no arrows below them in col 18
 *   Wave 2 — unblocked after wave 1:
 *     < at (2,8), < at (8,8)     → their left blocker is now gone
 *     > at (2,12), > at (8,12)   → their right blocker is now gone
 *
 * Column arrows (col 2, col 18) never share their column with any row arrow. ✓
 */
object TestLevel {

    // 11 rows × 20 cols
    // Row arrows at cols 4, 8, 12, 16  — never overlap col 2 or col 18
    // Col arrows at col 2 (^), col 18 (v)
    private val mazeLayout = listOf(
        "####################",  // row 0
        "#                  #",  // row 1
        "#   <   <   >   >  #",  // row 2: < at 4,8 ; > at 12,16
        "#                  #",  // row 3
        "# ^               v#",  // row 4: ^ at col 2, v at col 18
        "#                  #",  // row 5
        "# ^               v#",  // row 6: ^ at col 2, v at col 18
        "#                  #",  // row 7
        "#   <   <   >   >  #",  // row 8: < at 4,8 ; > at 12,16
        "#                  #",  // row 9
        "####################"   // row 10
    )

    fun build(): MazeState {
        val rows = mazeLayout.size
        val cols = mazeLayout.maxOf { it.length }

        val grid = mazeLayout.mapIndexed { r, rowStr ->
            List(cols) { c ->
                val ch = rowStr.getOrElse(c) { '#' }
                when (ch) {
                    '#' -> Cell(isWall = true)
                    '>' -> Cell(isWall = false, arrow = Direction.RIGHT)
                    '<' -> Cell(isWall = false, arrow = Direction.LEFT)
                    '^' -> Cell(isWall = false, arrow = Direction.UP)
                    'v' -> Cell(isWall = false, arrow = Direction.DOWN)
                    else -> Cell(isWall = false, arrow = null)
                }
            }
        }

        return MazeState(grid)
    }
}
