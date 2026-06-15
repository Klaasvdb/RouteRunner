package com.example.arrowremoval.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.drawscope.withTransform
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.arrowremoval.model.*
import kotlin.math.floor

private val BG_COLOR = Color(0xFF1a1a2e)
private val WALL_COLOR = Color(0xFF1a1a2e)
private val CORRIDOR_COLOR = Color(0xFF2a2a4e)
private val ARROW_COLOR = Color(0xFFa0a0d0)
private val REMOVED_DOT_COLOR = Color(0xFF3a3a5e)
private val INVALID_COLOR = Color(0xFFff4444)
private val SLITHER_COLOR = Color(0xFF5080ff)
private val WIN_COLOR = Color(0xFF44ff88)

@Composable
fun MazeScreen(viewModel: GameViewModel = viewModel()) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(BG_COLOR),
        contentAlignment = Alignment.Center
    ) {
        if (uiState.isComplete) {
            WinOverlay { viewModel.reset() }
        } else {
            MazeCanvas(
                uiState = uiState,
                onCellTapped = viewModel::onCellTapped,
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(
                        uiState.maze.cols.toFloat() / uiState.maze.rows.toFloat()
                    )
                    .padding(16.dp)
            )
        }
    }
}

@Composable
private fun WinOverlay(onReset: () -> Unit) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "All Cleared!",
            color = WIN_COLOR,
            fontSize = 36.sp,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onReset) {
            Text("Play Again")
        }
    }
}

@Composable
private fun MazeCanvas(
    uiState: GameUiState,
    onCellTapped: (Int, Int) -> Unit,
    modifier: Modifier = Modifier
) {
    val maze = uiState.maze

    Canvas(
        modifier = modifier.pointerInput(Unit) {
            detectTapGestures { offset ->
                val cellW = size.width.toFloat() / maze.cols
                val cellH = size.height.toFloat() / maze.rows
                val col = floor(offset.x / cellW).toInt().coerceIn(0, maze.cols - 1)
                val row = floor(offset.y / cellH).toInt().coerceIn(0, maze.rows - 1)
                onCellTapped(row, col)
            }
        }
    ) {
        val cellW = size.width / maze.cols
        val cellH = size.height / maze.rows

        // Draw corridors (non-wall cells get a filled background)
        for (r in 0 until maze.rows) {
            for (c in 0 until maze.cols) {
                val cell = maze.cell(r, c)
                if (!cell.isWall) {
                    drawRect(
                        color = CORRIDOR_COLOR,
                        topLeft = Offset(c * cellW, r * cellH),
                        size = Size(cellW, cellH)
                    )
                    // Fill gaps between adjacent corridors for seamless paths
                    if (r + 1 < maze.rows && !maze.cell(r + 1, c).isWall) {
                        drawRect(
                            color = CORRIDOR_COLOR,
                            topLeft = Offset(c * cellW, r * cellH + cellH * 0.5f),
                            size = Size(cellW, cellH)
                        )
                    }
                    if (c + 1 < maze.cols && !maze.cell(r, c + 1).isWall) {
                        drawRect(
                            color = CORRIDOR_COLOR,
                            topLeft = Offset(c * cellW + cellW * 0.5f, r * cellH),
                            size = Size(cellW, cellH)
                        )
                    }
                }
            }
        }

        // Draw maze border
        drawRect(
            color = ARROW_COLOR.copy(alpha = 0.3f),
            topLeft = Offset.Zero,
            size = size,
            style = Stroke(width = 2f)
        )

        // Draw arrows and removed dots
        for (r in 0 until maze.rows) {
            for (c in 0 until maze.cols) {
                val cell = maze.cell(r, c)
                if (cell.isWall) continue

                val cx = c * cellW + cellW / 2
                val cy = r * cellH + cellH / 2

                val isInvalid = uiState.invalidTapPos == GridPos(r, c)
                val isAnimating = uiState.animation?.let {
                    it.startRow == r && it.startCol == c
                } ?: false

                when {
                    cell.isRemoved && !isAnimating -> {
                        // Draw a small dot for removed arrows
                        drawCircle(
                            color = REMOVED_DOT_COLOR,
                            radius = cellW * 0.08f,
                            center = Offset(cx, cy)
                        )
                    }
                    cell.arrow != null && !cell.isRemoved -> {
                        val color = when {
                            isInvalid -> INVALID_COLOR
                            else -> ARROW_COLOR
                        }
                        drawArrow(cx, cy, cell.arrow, cellW, cellH, color)
                    }
                }
            }
        }

        // Draw slither animation
        uiState.animation?.let { anim ->
            drawSlitherAnimation(anim, cellW, cellH)
        }
    }
}

private fun DrawScope.drawSlitherAnimation(
    anim: SlitherAnimation,
    cellW: Float,
    cellH: Float
) {
    val path = anim.path
    if (path.isEmpty()) return

    val totalCells = path.size
    // How many cells the arrow has traveled through
    val cellProgress = anim.progress * totalCells

    // Tail = origin, head = leading edge moving outward
    val headIndex = cellProgress.toInt().coerceIn(0, totalCells - 1)
    val headFrac = cellProgress - headIndex

    // Draw the slither trail (already-passed cells)
    for (i in 0..headIndex) {
        val pos = path[i]
        val alpha = 1f - (i.toFloat() / totalCells)
        drawRect(
            color = SLITHER_COLOR.copy(alpha = alpha * 0.6f),
            topLeft = Offset(pos.col * cellW, pos.row * cellH),
            size = Size(cellW, cellH)
        )
    }

    // Draw the arrow head at its current animated position
    if (headIndex < path.size) {
        val pos = path[headIndex]
        val nextPos = if (headIndex + 1 < path.size) path[headIndex + 1] else null

        val baseCx = pos.col * cellW + cellW / 2
        val baseCy = pos.row * cellH + cellH / 2

        val cx: Float
        val cy: Float
        if (nextPos != null) {
            val nextCx = nextPos.col * cellW + cellW / 2
            val nextCy = nextPos.row * cellH + cellH / 2
            cx = baseCx + (nextCx - baseCx) * headFrac
            cy = baseCy + (nextCy - baseCy) * headFrac
        } else {
            cx = baseCx
            cy = baseCy
        }

        drawArrow(cx, cy, anim.direction, cellW, cellH, SLITHER_COLOR)
    }
}

private fun DrawScope.drawArrow(
    cx: Float, cy: Float,
    direction: Direction,
    cellW: Float, cellH: Float,
    color: Color
) {
    val arrowLen = minOf(cellW, cellH) * 0.55f
    val headLen = minOf(cellW, cellH) * 0.22f
    val strokeW = minOf(cellW, cellH) * 0.12f

    val degrees = when (direction) {
        Direction.RIGHT -> 0f
        Direction.DOWN -> 90f
        Direction.LEFT -> 180f
        Direction.UP -> 270f
    }

    withTransform({
        rotate(degrees = degrees, pivot = Offset(cx, cy))
    }) {
        // Arrow shaft
        drawLine(
            color = color,
            start = Offset(cx - arrowLen / 2, cy),
            end = Offset(cx + arrowLen / 2, cy),
            strokeWidth = strokeW,
            cap = StrokeCap.Round
        )
        // Arrowhead (two lines forming a ">")
        drawLine(
            color = color,
            start = Offset(cx + arrowLen / 2, cy),
            end = Offset(cx + arrowLen / 2 - headLen, cy - headLen * 0.7f),
            strokeWidth = strokeW,
            cap = StrokeCap.Round
        )
        drawLine(
            color = color,
            start = Offset(cx + arrowLen / 2, cy),
            end = Offset(cx + arrowLen / 2 - headLen, cy + headLen * 0.7f),
            strokeWidth = strokeW,
            cap = StrokeCap.Round
        )
    }
}
