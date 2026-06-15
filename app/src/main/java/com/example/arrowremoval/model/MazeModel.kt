package com.example.arrowremoval.model

enum class Direction { UP, DOWN, LEFT, RIGHT }

// A cell in the maze grid. `arrow` is null for wall or empty corridor cells.
data class Cell(
    val isWall: Boolean,
    val arrow: Direction? = null,
    val isRemoved: Boolean = false
)

data class GridPos(val row: Int, val col: Int)

// Represents an in-flight removal animation.
data class SlitherAnimation(
    val path: List<GridPos>,  // cells the arrow passes through during removal
    val direction: Direction,
    val startRow: Int,
    val startCol: Int,
    var progress: Float = 0f  // 0.0 to 1.0
)

class MazeState(val grid: List<List<Cell>>) {

    val rows get() = grid.size
    val cols get() = grid[0].size

    fun cell(row: Int, col: Int): Cell = grid[row][col]

    // Returns the straight-line path from (row, col) outward in arrow's direction
    // (the cells the arrow would travel through when removed, not including origin).
    private fun exitPath(row: Int, col: Int, dir: Direction): List<GridPos> {
        val path = mutableListOf<GridPos>()
        var r = row
        var c = col
        while (true) {
            val (dr, dc) = dir.delta()
            r += dr
            c += dc
            if (r < 0 || r >= rows || c < 0 || c >= cols) break
            path.add(GridPos(r, c))
        }
        return path
    }

    // An arrow is removable if no non-removed arrow exists in front of it
    // (straight line in arrow's direction, to the grid edge).
    fun canRemove(row: Int, col: Int): Boolean {
        val arrow = grid[row][col].arrow ?: return false
        if (grid[row][col].isRemoved) return false
        val path = exitPath(row, col, arrow)
        return path.none { (r, c) ->
            val cell = grid[r][c]
            !cell.isWall && cell.arrow != null && !cell.isRemoved
        }
    }

    fun buildSlither(row: Int, col: Int): SlitherAnimation? {
        val arrow = grid[row][col].arrow ?: return null
        val path = exitPath(row, col, arrow)
        // Include origin cell in path so animation starts there
        val fullPath = listOf(GridPos(row, col)) + path
        return SlitherAnimation(fullPath, arrow, row, col)
    }

    // Apply a removal — returns a new MazeState with the cell marked removed.
    fun withRemoved(row: Int, col: Int): MazeState {
        val newGrid = grid.mapIndexed { r, rowList ->
            rowList.mapIndexed { c, cell ->
                if (r == row && c == col) cell.copy(isRemoved = true) else cell
            }
        }
        return MazeState(newGrid)
    }

    fun isComplete(): Boolean = grid.all { row ->
        row.all { cell -> cell.isWall || cell.arrow == null || cell.isRemoved }
    }
}

private fun Direction.delta(): Pair<Int, Int> = when (this) {
    Direction.UP -> -1 to 0
    Direction.DOWN -> 1 to 0
    Direction.LEFT -> 0 to -1
    Direction.RIGHT -> 0 to 1
}
